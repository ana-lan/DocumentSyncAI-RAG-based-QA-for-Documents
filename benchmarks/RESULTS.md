# Serving Benchmarks: vLLM on GKE (NVIDIA L4)

Measured on Oct 10, 2026. Every number below comes from the raw result files in `benchmarks/results/`.

## Key results

- **Continuous batching:** output throughput scaled **17x** (17 to 298 tokens/s) from 1 to 32 concurrent requests on a single L4.
- **Prefix caching:** on prompts sharing a 768-token prefix, p95 time-to-first-token dropped **67%** (7.3 s to 2.4 s) and throughput rose **62%** at 32 concurrent requests.
- **FP8 quantization:** weight memory dropped **43%**, KV cache capacity grew **2.5x**, throughput rose **49 to 63%**, and end-to-end RAG p95 latency dropped **38%**, with **no accuracy loss** on a 20-question eval.
- **Pipeline fix:** caching the embedding model cut retrieval p50 latency from **1.04 s to 9 ms**.

## Setup

| Component | Details |
|---|---|
| Cluster | GKE Standard, zonal (us-central1-a), Kubernetes 1.35 |
| CPU node | e2-standard-4 (DocumentSync app) |
| GPU node | g2-standard-8 with 1x NVIDIA L4, 24 GB |
| Inference server | vLLM 0.31.0 (V1 engine, FlashAttention 2, CUDA graphs) |
| Model | Qwen/Qwen2.5-7B-Instruct, max context 8,192, `--gpu-memory-utilization=0.90` |
| App | DocumentSync RAG (pdfplumber, fixed 1,024-character chunks, all-MiniLM-L6-v2, ChromaDB), image `app:v2` |
| Networking | vLLM as internal ClusterIP service; app public via LoadBalancer |
| Autoscaling | HorizontalPodAutoscaler on the app, 1 to 3 replicas at 70% CPU |

The app reaches the LLM through an OpenAI-compatible client configured by environment variables (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`). The same image runs against Groq locally and against vLLM in the cluster with no code changes.

## Methodology

- **Server load tests:** `vllm bench serve` run inside the vLLM pod, so network latency is excluded.
  - Synthetic random prompts, 1,024 input tokens and 256 output tokens (`--ignore-eos` fixes output length).
  - `--max-concurrency` holds a fixed number of requests in flight.
  - Each concurrency level uses a different seed, so later runs cannot hit prefix-cache entries from earlier runs.
- **Prefix caching test:** every prompt shares a 768-token prefix plus 256 unique tokens, with 128 output tokens. ON and OFF used identical prompts (seeds 101 and 132).
- **End-to-end test (`benchmarks/e2e.py`):** 20 factual questions about "Attention Is All You Need", run sequentially through the real pipeline (retrieve, then generate) from inside the app pod.
  - Retrieval and generation are timed separately.
  - 2 warmup questions are excluded from the results.
  - Each answer is saved next to the paper's ground truth for a quality check.
- **Metrics:**
  - TTFT: time to first token
  - TPOT: time per output token, after the first
  - E2E latency: total time per request
  - Output throughput: generated tokens per second across all requests

## 1. Concurrency sweep (bf16)

| Concurrency | Output tok/s | vs c=1 | TTFT p50 / p95 | TPOT p50 | E2E latency p50 / p95 |
|---|---|---|---|---|---|
| 1 | 17.2 | 1x | 304 / 310 ms | 57 ms | 14.9 / 14.9 s |
| 8 | 115.5 | 6.7x | 977 / 2,004 ms | 66 ms | 17.7 / 18.0 s |
| 32 | 297.7 | 17.3x | 1,938 / 7,314 ms | 99 ms | 27.3 / 32.6 s |

**Findings**

- **One request at a time is limited by memory bandwidth.** Every generated token requires reading all 14.3 GB of weights, and the L4 reads about 300 GB/s. That caps a single request at about 17 tokens/s, which is close to the measured value (TPOT 57 ms).
- **Batching amortizes those weight reads across requests.** At c=8, throughput rises 6.7x while per-token latency rises only 15%.
- **At c=32, throughput keeps climbing (17x), but p95 TTFT exceeds 7 s.** New prompts have to wait for prefill time while 31 other requests are decoding. The p95 inter-token latency of 342 ms reflects these prefill interruptions.

## 2. Prefix caching ON vs OFF (bf16)

| Metric | OFF | ON | Change |
|---|---|---|---|
| c=1 TTFT p50 | 298 ms | 110 ms | 63% lower |
| c=1 TTFT p95 | 309 ms | 120 ms | 61% lower |
| c=32 TTFT p50 | 1,914 ms | 1,490 ms | 22% lower |
| c=32 TTFT p95 | 7,302 ms | 2,381 ms | 67% lower |
| c=32 output throughput | 227 tok/s | 367 tok/s | +62% |
| c=32 TPOT p50 | 125 ms | 76 ms | 39% lower |
| c=32 E2E latency p95 | 23.1 s | 11.6 s | 50% lower |

**Findings**

- **With caching, vLLM reuses the stored KV blocks for the shared 768 tokens** and only prefills the 256 unique ones. At c=1 that cuts TTFT by about two thirds.
- **Under load the effect compounds.** Removing repeated prefill frees GPU time for decoding, which lowers tail latency and raises throughput.
- **Decode speed at c=1 is unchanged (57 ms TPOT both ways),** as expected: caching only affects prefill.
- **On DocumentSync's real prompts the effect was within run-to-run noise** (see section 4). Retrieved chunks differ per question, so the only shared prefix is Qwen's roughly 30-token default system prompt.

## 3. FP8 vs bf16

Same workload and seeds as the bf16 sweep. Prefix caching is on (the default) in both.

| Metric | bf16 | FP8 | Change |
|---|---|---|---|
| Weight memory | 14.29 GiB | 8.17 GiB | 43% smaller |
| KV cache capacity | 72,464 tokens | 184,160 tokens | 2.5x |
| Max concurrency at full 8K context | 8.85x | 22.48x | 2.5x |
| c=1 output tok/s | 17.2 | 28.2 | +63% |
| c=1 TPOT p50 | 57 ms | 35 ms | 39% lower |
| c=1 TTFT p50 | 304 ms | 215 ms | 29% lower |
| c=1 E2E latency p50 | 14.9 s | 9.1 s | 39% lower |
| c=32 output tok/s | 297.7 | 444.8 | +49% |
| c=32 TTFT p50 / p95 | 1,938 / 7,314 ms | 1,354 / 5,490 ms | 30% / 25% lower |
| c=32 TPOT p50 | 99 ms | 66 ms | 33% lower |
| c=32 E2E latency p50 / p95 | 27.3 / 32.6 s | 18.4 / 22.3 s | 33% / 32% lower |

**Findings**

- **FP8 attacks the bandwidth bottleneck from section 1 directly.** Fewer bytes per token means faster decoding.
- **The memory it frees becomes KV cache,** so the same GPU holds 2.5x more concurrent context.
- **Weights shrink by 43% rather than 50%** because the embedding and output layers (152K-token vocabulary) stay in bf16.

## 4. End-to-end RAG latency (20 real questions, sequential)

| Configuration | Retrieval p50 | Total p50 | Total p95 | Mean | Correct |
|---|---|---|---|---|---|
| bf16, prefix caching ON | 24 ms | 1.59 s | 4.36 s | 1.92 s | 17/20 |
| bf16, prefix caching OFF | 28 ms | 1.71 s | 5.54 s | 2.12 s | 17/20 |
| FP8, prefix caching ON | 21 ms | **1.00 s** | **2.71 s** | 1.22 s | 17/20 |

**Findings**

- **FP8 lowers real end-to-end latency by about 37%** (p50 and p95) with identical accuracy.
- **Answer length drives latency.** Short factual answers return in about 1 s; long explanations take 3 to 7 s.
- **All 3 misses were retrieval failures in every configuration.** The model answered "I don't know" instead of guessing. The cause is PDF text extraction dropping spaces between words.
- **The caching ON vs OFF gap here is mostly answer-length variance.** The app samples at temperature 0.2, and one long answer moved the p95.

## 5. Retrieval fix

`get_embedding_model()` reloaded the sentence-transformer from disk on every query. The app's latency display only timed the LLM call, so the cost was invisible. Adding `@lru_cache` loads the model once per process.

| Retrieval latency | Before | After |
|---|---|---|
| p50 (local, Apple Silicon) | 1.036 s | 0.009 s |
| p95 (local) | 1.383 s | 0.024 s |
| p50 / p95 (GKE e2-standard-4) | n/a | 20 ms / 22 ms |

Before the fix, retrieval took longer than the LLM call itself on a fast backend. After the fix it is about 1.5% of end-to-end latency.

## Operational notes

- **Spot preemption.** The spot L4 node was reclaimed about 20 minutes after creation. The Deployment rescheduled vLLM automatically, but a cold start on a fresh node (driver install, image pull, weight download, compile) took about 15 minutes. Benchmarks were moved to an on-demand node for a stable window.
- **Autoscaling observed.** Running the e2e benchmark inside the app pod pushed CPU past the 70% target. The HPA scaled the app from 1 to 3 replicas, then back to 1 after its stabilization window.
- **Env var collision.** Kubernetes injects service discovery variables named after each Service. A Service named `vllm` produced `VLLM_PORT=tcp://...`, which vLLM reads as its own config and crashes on. The fix is `enableServiceLinks: false` in the pod spec.
- **Single-GPU rollouts.** The default RollingUpdate starts the new pod before stopping the old one, which deadlocks when there is only one GPU. The fix is `strategy: Recreate`.
- **Startup timing (bf16):**
  - weight download about 90 s
  - model load 125 s
  - torch.compile 27 s
  - about 4 minutes from pod start to ready on a node with the image cached

## Recommendations

1. **Serve with FP8 on the L4.** It gives 49 to 63% higher throughput, 2.5x KV cache capacity, and about 37% lower end-to-end latency, with no accuracy loss on this eval. Validate on a larger eval set before production.
2. **For interactive traffic, target about 8 to 16 concurrent requests per replica** to keep TTFT near 2 s or less, and add replicas beyond that. Push concurrency higher only for offline batch work.
3. **Keep prefix caching on, and design prompts to share prefixes.** Put stable content (system instructions, document summaries, context reused across follow-up questions on the same document) at the start of the prompt.
4. **Use on-demand or reserved GPUs for latency-sensitive serving,** since a preemption costs about 15 minutes of cold start. Spot fits batch jobs that tolerate restarts. A persistent model cache volume would cut restart time.
5. **Fix PDF text extraction and drop the references section before chunking.** For extraction, tune pdfplumber's `x_tolerance` so spaces between words are kept. Both changes target the 3/20 retrieval misses, which are now the main accuracy limit.

## Limitations

- Each configuration was measured in a single run, with no repeated trials or confidence intervals.
- Server load tests use synthetic random prompts rather than real traffic.
- The quality eval is 20 questions on one paper: enough to catch obvious regressions, not to prove equivalence.
- The end-to-end test measures total latency, not TTFT, so small prefill savings are hidden behind generation time.
- Sampling (temperature 0.2) causes small run-to-run variation in answer length.

## Reproduce

Deploy (see `k8s/`):

```bash
kubectl apply -f k8s/vllm.yaml
kubectl apply -f k8s/docsync.yaml
```

`k8s/vllm.yaml` ships with `--quantization=fp8` (the recommended config):
- Remove that line for the bf16 baseline.
- Add `--no-enable-prefix-caching` for the caching-off runs.

Server benchmark, run inside the vLLM pod (`kubectl exec -it deploy/vllm -- bash`):

```bash
# Concurrency sweep: C in {1, 8, 32}, prompts N in {20, 80, 160}, seed = C
vllm bench serve --backend vllm --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset-name random --random-input-len 1024 --random-output-len 256 --ignore-eos \
  --num-prompts $N --max-concurrency $C --seed $C \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
  --save-result --result-dir /tmp/bench --result-filename sweep_c${C}.json

# Prefix caching: C in {1, 32}, N in {20, 160}, seed = 100 + C
vllm bench serve --backend vllm --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset-name random --random-prefix-len 768 --random-input-len 256 --random-output-len 128 --ignore-eos \
  --num-prompts $N --max-concurrency $C --seed $((100+C)) \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
  --save-result --result-dir /tmp/bench --result-filename prefix_on_c${C}.json
```

End-to-end, run from the app pod:

```bash
kubectl exec -it deploy/docsync -- python -m benchmarks.e2e --label vllm_l4
```

Locally, against any OpenAI-compatible backend:

```bash
python -m benchmarks.e2e --label local --limit 5 --pause 20
python -m benchmarks.e2e --label retrieval_only --retrieval-only
```

## Raw data

| Files | Contents |
|---|---|
| `benchmarks/results/gke/vllm_cache_on/sweep_c{1,8,32}.json` | bf16 concurrency sweep |
| `benchmarks/results/gke/vllm_cache_on/prefix_on_c{1,32}.json` | prefix caching ON |
| `benchmarks/results/gke/vllm_cache_off/prefix_off_c{1,32}.json` | prefix caching OFF |
| `benchmarks/results/gke/vllm_fp8/fp8_c{1,32}.json` | FP8 sweep |
| `benchmarks/results/gke/e2e_vllm_l4_{cache_on,cache_off,fp8}.json` | end-to-end runs, with every answer |
| `benchmarks/results/gke/e2e_gke_retrieval.json` | retrieval latency on the GKE node |
| `benchmarks/results/e2e_retrieval_{before,after}_fix.json` | retrieval fix, measured locally |