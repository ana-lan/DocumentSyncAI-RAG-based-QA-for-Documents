"""
E0: predict what one NVIDIA L4 can physically do for Qwen2.5-7B, then compare
with what vLLM measured.

Two limits, one for each phase of a request:

  Decode (one new token per step) is MEMORY-BOUND. Each step reads every weight
  once from GPU memory, plus the KV cache of every sequence in the batch. The
  fastest a step can go is  bytes_read / memory_bandwidth.

  Prefill (reading the prompt) is COMPUTE-BOUND. Every prompt token goes through
  every weight: about 2 FLOPs per parameter per token. The fastest it can go is
  flops / peak_tensor_flops.

The ratio predicted/measured says how close vLLM gets to the hardware ceiling.
The rest is overhead that E4's profile will try to explain.

Usage (from repo root):
  python -m benchmarks.perf_model
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WEEK1 = os.path.join(HERE, "results", "gke")

# NVIDIA L4 spec sheet: https://www.nvidia.com/en-us/data-center/l4/
# The sheet lists tensor-core FLOPS "with sparsity"; dense math (what vLLM runs) is half.
L4 = {
    "bandwidth_gb_s": 300,
    "tflops_dense": {"bf16": 242 / 2, "fp8": 485 / 2},
}

# Qwen2.5-7B-Instruct architecture (config.json on Hugging Face)
QWEN = {
    "layers": 28,
    "hidden": 3584,
    "intermediate": 18944,
    "kv_heads": 4,       # grouped-query attention: 28 query heads share 4 K/V heads
    "head_dim": 128,
    "vocab": 152064,
}


def param_counts(m: dict = QWEN) -> dict:
    """Parameters per part of the model. Biases and norms are under 0.01%, so ignored."""
    h, kv = m["hidden"], m["kv_heads"] * m["head_dim"]
    attention = 2 * h * h + 2 * h * kv          # q, o projections + k, v projections
    mlp = 3 * h * m["intermediate"]             # gate, up, down projections
    return {
        "layers": m["layers"] * (attention + mlp),   # the transformer blocks
        "embed": m["vocab"] * h,                     # input lookup table
        "lm_head": m["vocab"] * h,                   # output projection (not tied in Qwen2.5-7B)
    }


def weight_bytes_per_step(precision: str) -> float:
    """
    Bytes of weights read to produce one token.

    The input embedding is a lookup: only one row per token is read, so it is left out.
    vLLM's fp8 mode quantizes the linear layers in the blocks to 1 byte, but keeps
    the lm_head (and embeddings) in bf16, 2 bytes.
    """
    p = param_counts()
    layer_bytes = 1 if precision == "fp8" else 2
    return p["layers"] * layer_bytes + p["lm_head"] * 2


def kv_bytes_per_token(m: dict = QWEN, kv_dtype_bytes: int = 2) -> int:
    """KV cache stored per token of context: K and V, for every layer, for each KV head. bf16 by default."""
    return 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * kv_dtype_bytes


def decode_step_ms(precision: str, batch: int, avg_context: float) -> dict:
    """
    Fastest possible time for one decode step with `batch` sequences in flight.

    Memory time: weights once (shared by the whole batch) + each sequence's KV cache.
    Compute time: 2 FLOPs per weight per sequence. Whichever is larger is the bound.
    """
    p = param_counts()
    bytes_read = weight_bytes_per_step(precision) + batch * avg_context * kv_bytes_per_token()
    memory_ms = bytes_read / (L4["bandwidth_gb_s"] * 1e9) * 1e3
    flops = 2 * (p["layers"] + p["lm_head"]) * batch
    compute_ms = flops / (L4["tflops_dense"][precision] * 1e12) * 1e3
    return {
        "step_ms": max(memory_ms, compute_ms),
        "memory_ms": memory_ms,
        "compute_ms": compute_ms,
        "bound": "memory" if memory_ms >= compute_ms else "compute",
    }


def prefill_ms(precision: str, prompt_tokens: int) -> float:
    """
    Fastest possible time to process a prompt: weight matmuls + attention scores.
    Attention is causal, so each token only looks at earlier tokens (the /2).
    """
    m, p = QWEN, param_counts()
    linear = 2 * p["layers"] * prompt_tokens
    attention = 2 * 2 * m["layers"] * prompt_tokens ** 2 * m["hidden"] / 2   # QK^T and attn x V
    return (linear + attention) / (L4["tflops_dense"][precision] * 1e12) * 1e3


# Week 1 runs: 1,024 input tokens, 256 output tokens
WEEK1_RUNS = [
    ("bf16", "vllm_cache_on/sweep_c1.json"),
    ("bf16", "vllm_cache_on/sweep_c8.json"),
    ("bf16", "vllm_cache_on/sweep_c32.json"),
    ("fp8", "vllm_fp8/fp8_c1.json"),
    ("fp8", "vllm_fp8/fp8_c32.json"),
]
INPUT_LEN, OUTPUT_LEN = 1024, 256


def compare(precision: str, run: dict) -> dict:
    c = run["max_concurrency"]
    avg_context = INPUT_LEN + OUTPUT_LEN / 2          # context grows from 1024 to 1280 during decode
    step = decode_step_ms(precision, c, avg_context)
    return {
        "precision": precision,
        "concurrency": c,
        "bound": step["bound"],
        "pred_step_ms": round(step["step_ms"], 1),
        "meas_itl_p50_ms": round(run["median_itl_ms"], 1),
        "meas_tpot_p50_ms": round(run["median_tpot_ms"], 1),
        # efficiency: how much of the hardware ceiling vLLM reached (100% = at the ceiling)
        "eff_vs_itl": round(step["step_ms"] / run["median_itl_ms"], 3),
        "eff_vs_tpot": round(step["step_ms"] / run["median_tpot_ms"], 3),
        "pred_ceiling_tok_s": round(c / step["step_ms"] * 1e3, 1),
        "meas_output_tok_s": round(run["output_throughput"], 1),
        "pred_prefill_ms": round(prefill_ms(precision, INPUT_LEN), 1),
        "meas_ttft_p50_ms": round(run["median_ttft_ms"], 1),
    }


def fit_overhead(rows: list[dict]) -> dict:
    """
    At c=1, assume step_time = bytes_read / bandwidth + fixed_overhead, the same for
    both precisions. bf16 and fp8 read different byte counts, so two measurements
    give two equations, and we can solve for both unknowns. It separates "memory is
    slower than the spec" from "something else costs a fixed few ms every step".
    """
    c1 = {r["precision"]: r for r in rows if r["concurrency"] == 1}
    ctx = INPUT_LEN + OUTPUT_LEN / 2
    b = {prec: weight_bytes_per_step(prec) + ctx * kv_bytes_per_token() for prec in c1}
    t = {prec: c1[prec]["meas_itl_p50_ms"] / 1e3 for prec in c1}
    bandwidth = (b["bf16"] - b["fp8"]) / (t["bf16"] - t["fp8"])
    overhead = t["bf16"] - b["bf16"] / bandwidth
    return {"bandwidth_gb_s": round(bandwidth / 1e9, 1), "overhead_ms": round(overhead * 1e3, 1)}


def main():
    p = param_counts()
    print("Qwen2.5-7B parameters: "
          f"layers {p['layers']/1e9:.3f}B, embed {p['embed']/1e9:.3f}B, lm_head {p['lm_head']/1e9:.3f}B, "
          f"total {sum(p.values())/1e9:.3f}B")
    for prec in ("bf16", "fp8"):
        print(f"{prec}: weights read per decode step {weight_bytes_per_step(prec)/1e9:.2f} GB | "
              f"c=1 ceiling {1e3/decode_step_ms(prec, 1, 0)['step_ms']:.1f} tok/s (weights only)")
    print(f"KV cache per token of context: {kv_bytes_per_token()/1024:.0f} KiB\n")

    rows = []
    for prec, path in WEEK1_RUNS:
        with open(os.path.join(WEEK1, path)) as f:
            rows.append(compare(prec, json.load(f)))

    header = (f"{'prec':<5}{'c':>3} {'bound':<7}{'pred step':>10}{'ITL p50':>9}{'eff':>6}"
              f"{'TPOT p50':>10}{'eff':>6}{'ceiling tok/s':>15}{'meas tok/s':>12}"
              f"{'pred prefill':>14}{'TTFT p50':>10}")
    print(header)
    for r in rows:
        print(f"{r['precision']:<5}{r['concurrency']:>3} {r['bound']:<7}{r['pred_step_ms']:>8} ms"
              f"{r['meas_itl_p50_ms']:>6} ms{r['eff_vs_itl']:>6.0%}{r['meas_tpot_p50_ms']:>7} ms{r['eff_vs_tpot']:>6.0%}"
              f"{r['pred_ceiling_tok_s']:>15}{r['meas_output_tok_s']:>12}"
              f"{r['pred_prefill_ms']:>11} ms{r['meas_ttft_p50_ms']:>7} ms")

    fit = fit_overhead(rows)
    print(f"\nc=1 fit (step = bytes / bandwidth + fixed overhead), using bf16 and fp8 together:")
    print(f"  effective bandwidth {fit['bandwidth_gb_s']} GB/s ({fit['bandwidth_gb_s']/L4['bandwidth_gb_s']:.0%} of spec), "
          f"fixed overhead {fit['overhead_ms']} ms per step")
    print("  E4's profile should show roughly that much non-memory time per decode step.")

    out = os.path.join(HERE, "results", "perf_model_week1.json")
    with open(out, "w") as f:
        json.dump({"hardware": L4, "model": QWEN, "params": p, "rows": rows, "c1_fit": fit}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()