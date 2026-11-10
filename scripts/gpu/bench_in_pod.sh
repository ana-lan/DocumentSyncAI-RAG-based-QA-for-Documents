#!/bin/bash
# Runs INSIDE the vLLM pod (run.sh copies it in). Same method as week 1:
# random prompts, fixed output length (--ignore-eos), a distinct seed per run so no
# run can reuse another's prefix cache, and the benchmark next to the server.
#
#   bash bench_in_pod.sh sweep    LABEL   # E3: c = 1..48, seed = c, N = max(20, 5c)
#   bash bench_in_pod.sh repeats  LABEL   # E3: c = 1, 8, 32, three times, fresh seeds
#   bash bench_in_pod.sh profile  LABEL   # E4: torch trace of steady decode at c = 1 and c = 32
#   bash bench_in_pod.sh long     LABEL   # E6: 8,000-token prompts at c = 1, 4, 8
#   bash bench_in_pod.sh spec     LABEL   # E7: week 1 workload at c = 1 and 32, plus acceptance counters
set -euo pipefail
MODE=$1
LABEL=$2
OUT=/tmp/bench/$LABEL
MODEL=Qwen/Qwen2.5-7B-Instruct
mkdir -p "$OUT"

# bench C N SEED INPUT_LEN OUTPUT_LEN FILE
bench() {
  local c=$1 n=$2 seed=$3 inlen=$4 outlen=$5 file=$6
  echo ">> c=$c n=$n seed=$seed in=$inlen out=$outlen -> $file"
  vllm bench serve --backend vllm --base-url http://localhost:8000 --model "$MODEL" \
    --dataset-name random --random-input-len "$inlen" --random-output-len "$outlen" --ignore-eos \
    --num-prompts "$n" --max-concurrency "$c" --seed "$seed" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --goodput ttft:2000 tpot:100 \
    --save-result --result-dir "$OUT" --result-filename "$file" > "$OUT/${file%.json}.log" 2>&1
  grep -E "Output token throughput|Median TTFT|P95 TTFT|Median TPOT|P95 TPOT|goodput" "$OUT/${file%.json}.log" || true
}

# POST to a vLLM endpoint (python, since the image may not ship curl)
post() {
  python3 -c "import urllib.request as u; u.urlopen(u.Request('http://localhost:8000/$1', method='POST'), timeout=900)"
}
metrics() {
  python3 -c "import urllib.request as u; print(u.urlopen('http://localhost:8000/metrics').read().decode())"
}

case "$MODE" in
  sweep)
    for c in 1 4 8 12 16 24 32 48; do
      n=$(( c * 5 > 20 ? c * 5 : 20 ))
      bench "$c" "$n" "$c" 1024 256 "sweep_c${c}.json"
    done ;;
  repeats)
    for rep in 1 2 3; do
      for c in 1 8 32; do
        n=$(( c == 1 ? 10 : c * 5 ))
        bench "$c" "$n" $(( 1000 * rep + c )) 1024 256 "rep${rep}_c${c}.json"
      done
    done ;;
  profile)
    # Start a run, wait until every request is past prefill (steady decode), then
    # trace a few seconds. The profiler only writes while started.
    for c in 1 32; do
      n=$(( c == 1 ? 4 : 64 ))
      bench "$c" "$n" $(( 900 + c )) 1024 256 "profile_c${c}.json" &
      sleep 30
      echo ">> profiling c=$c"
      post start_profile || echo "WARNING: start_profile failed (is --profiler-config set?)"
      sleep 4
      post stop_profile || echo "WARNING: stop_profile failed"
      wait
    done
    ls -la /profiles ;;
  long)
    for c in 1 4 8; do
      n=$(( c * 2 > 8 ? c * 2 : 8 ))
      bench "$c" "$n" $(( 500 + c )) 8000 256 "long_c${c}.json"
    done ;;
  spec)
    metrics | grep -i spec_decode > "$OUT/spec_metrics_before.txt" || true
    bench 1 10 701 1024 256 "spec_c1.json"
    bench 32 160 732 1024 256 "spec_c32.json"
    metrics | grep -i spec_decode > "$OUT/spec_metrics_after.txt" || true
    echo ">> speculative decoding counters (after):"; cat "$OUT/spec_metrics_after.txt" ;;
  *)
    echo "unknown mode $MODE"; exit 1 ;;
esac
echo "done: $OUT"