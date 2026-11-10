#!/bin/bash
# GPU-day steps, run from the laptop in the repo root. Results land in
# benchmarks/results/gke_week2/ as soon as each step finishes.
#
#   bash scripts/gpu/run.sh deploy bf16|fp8|fp8-32k|fp8-spec   # (re)start vLLM, then print the checks
#   bash scripts/gpu/run.sh flags                              # confirm --goodput, profiler, speculative flags
#   bash scripts/gpu/run.sh e3 bf16                            # E3 sweep + repeats (~35 min)
#   bash scripts/gpu/run.sh e4                                 # E4 profile (bf16 only)
#   bash scripts/gpu/run.sh e1 bf16                            # E1 quality run on the 120 questions
#   bash scripts/gpu/run.sh e2 bf16 "A B C"                    # E2 variants, 3 interleaved rounds
#   bash scripts/gpu/run.sh e6                                 # E6 long prompts (fp8-32k)
#   bash scripts/gpu/run.sh e7                                 # E7 speculative decoding (fp8-spec)
set -euo pipefail
MODEL=Qwen/Qwen2.5-7B-Instruct
DEST=benchmarks/results/gke_week2
mkdir -p "$DEST"

vllm_pod() { kubectl get pod -l app=vllm -o jsonpath='{.items[0].metadata.name}'; }
app_pod()  { kubectl get pod -l app=docsync -o jsonpath='{.items[0].metadata.name}'; }

in_pod() {   # in_pod MODE LABEL: run bench_in_pod.sh inside vLLM, copy results out
  local pod; pod=$(vllm_pod)
  kubectl cp scripts/gpu/bench_in_pod.sh "$pod:/tmp/bench_in_pod.sh"
  kubectl exec "$pod" -- bash /tmp/bench_in_pod.sh "$1" "$2"
  kubectl cp "$pod:/tmp/bench/$2" "$DEST/$2"
}

quality() {  # quality LABEL [extra quality_eval args]: run inside the app pod, copy the file out
  local label=$1; shift
  kubectl exec "$(app_pod)" -- python -m benchmarks.quality_eval --label "$label" \
    --base-url http://vllm:8000/v1 --model "$MODEL" "$@"
  kubectl cp "$(app_pod):/app/benchmarks/results/week2/quality_${label}.json" "$DEST/quality/quality_${label}.json"
}
mkdir -p "$DEST/quality"

case "$1" in
  deploy)
    kubectl apply -f "k8s/week2/vllm-$2.yaml"
    kubectl rollout status deploy/vllm --timeout=40m
    echo; echo "== checks: confirm these before running anything =="
    kubectl logs deploy/vllm | grep -iE "quantization|prefix.caching|KV cache size|Maximum concurrency|speculative|profiler|max_model_len" | head -20
    kubectl logs deploy/vllm > "$DEST/startup_$2.log"
    ;;
  flags)
    kubectl exec "$(vllm_pod)" -- bash -c \
      "vllm bench serve --help=all 2>/dev/null | grep -E -- '--goodput|--percentile-metrics' ; \
       vllm serve --help=all 2>/dev/null | grep -E -- '--profiler-config|--speculative-config'"
    ;;
  e3)
    in_pod sweep "e3_$2_sweep"
    in_pod repeats "e3_$2_repeats"
    ;;
  e4)
    in_pod profile e4_bf16
    kubectl cp "$(vllm_pod):/profiles" "$DEST/e4_bf16/traces"
    ;;
  e1)
    quality "e1_$2" --concurrency 8
    ;;
  e2)
    # Interleave A, B, C in each round so slow drift in the server can't favour one variant.
    for round in 1 2 3; do
      for v in $3; do
        case $v in
          A) flags="--layout context_first --order score" ;;
          B) flags="--layout instructions_first --order score" ;;
          C) flags="--layout instructions_first --order position" ;;
        esac
        quality "e2_$2_${v}_r${round}" --input eval/week2/frozen_sessions.json $flags --reset-cache
      done
    done
    ;;
  e6)
    in_pod long e6_fp8_32k
    ;;
  e7)
    in_pod spec e7_fp8_spec
    quality e7_fp8_spec_real_c1 --limit 30   # real RAG prompts: draft acceptance is more realistic than on random tokens
    ;;
  *)
    echo "unknown step $1"; exit 1 ;;
esac