"""
Phase 7: turn the week 2 GPU results into tables (for RESULTS.md) and charts (for the post).

Reads benchmarks/results/gke_week2/ and prints one section per experiment:
  E3  capacity under the 2 s TTFT / 100 ms TPOT target, plus run-to-run spread
  E0  predicted ceiling vs measured, for every E3 point, and the refit overhead
  E4  where a decode step's time goes, from the torch profiler traces
  E2  prefix-cache hit rate and TTFT per prompt variant
  E6  long prompts at 32K context
  E7  n-gram speculative decoding vs FP8 alone, random vs real RAG prompts
Saves everything to gke_week2/analysis.json and charts to gke_week2/charts/.

Usage (from repo root):
  python -m benchmarks.analyze_week2
"""
import glob
import gzip
import json
import os
import re
import statistics as st
from collections import defaultdict

from benchmarks.perf_model import compare, decode_step_ms, fit_overhead, L4

R = "benchmarks/results/gke_week2"
OUT = {}
TTFT_TARGET_MS, TPOT_TARGET_MS = 2000, 100

# Week 1 baseline (benchmarks/RESULTS.md, Oct 10, same vLLM 0.31.0, same L4)
WEEK1 = {"bf16": {1: 17.2, 32: 297.7}, "fp8": {1: 28.2, 32: 444.8}}

# E7 counters right after the 30 real RAG prompts are saved in e7_fp8_spec/spec_metrics_after_real.txt
# (the "after benchmark" values are in spec_metrics_after.txt). Real-prompt counts = the difference.


def load(path):
    with open(path) as f:
        return json.load(f)


def runs(folder):
    """{concurrency: run json} for every vllm bench result in a folder."""
    out = {}
    for p in glob.glob(os.path.join(R, folder, "*.json")):
        d = load(p)
        out.setdefault(d["max_concurrency"], []).append(d)
    return out


def row(d):
    good = d.get("request_goodput")
    rt = d.get("request_throughput") or 0
    return {
        "c": d["max_concurrency"], "n": d.get("completed"),
        "out_tok_s": round(d["output_throughput"], 1),
        "ttft_p50": round(d["median_ttft_ms"]), "ttft_p95": round(d["p95_ttft_ms"]),
        "tpot_p50": round(d["median_tpot_ms"], 1), "tpot_p95": round(d["p95_tpot_ms"], 1),
        "itl_p50": round(d["median_itl_ms"], 1),
        "goodput_share": round(good / rt, 2) if good is not None and rt else None,
        "meets": d["p95_ttft_ms"] <= TTFT_TARGET_MS and d["p95_tpot_ms"] <= TPOT_TARGET_MS,
    }


# ── E3 ────────────────────────────────────────────────────
def e3():
    print("\n" + "=" * 70 + "\nE3  capacity under p95 TTFT <= 2 s and p95 TPOT <= 100 ms\n" + "=" * 70)
    res = {}
    for prec in ("bf16", "fp8"):
        sweep = runs(f"e3_{prec}_sweep")
        rows = [row(sweep[c][0]) for c in sorted(sweep)]
        print(f"\n{prec}")
        print(f"{'c':>3}{'tok/s':>9}{'TTFT p50':>10}{'TTFT p95':>10}{'TPOT p50':>10}{'TPOT p95':>10}{'goodput':>9}  target")
        for r in rows:
            g = f"{r['goodput_share']:.0%}" if r["goodput_share"] is not None else "n/a"
            print(f"{r['c']:>3}{r['out_tok_s']:>9}{r['ttft_p50']:>10}{r['ttft_p95']:>10}{r['tpot_p50']:>10}"
                  f"{r['tpot_p95']:>10}{g:>9}  {'yes' if r['meets'] else 'NO'}")
        ok = [r for r in rows if r["meets"]]
        best = max(ok, key=lambda r: r["c"]) if ok else None
        if best:
            print(f"-> highest concurrency meeting the target: c={best['c']} at {best['out_tok_s']} tok/s")

        reps = runs(f"e3_{prec}_repeats")
        spread = {}
        print(f"\n{prec} repeats (3 runs, fresh seeds): median [min to max]")
        for c in sorted(reps):
            vals = {k: [row(d)[k] for d in reps[c]] for k in ("out_tok_s", "ttft_p95", "tpot_p50")}
            spread[c] = {k: {"median": st.median(v), "min": min(v), "max": max(v)} for k, v in vals.items()}
            line = "  ".join(f"{k} {st.median(v)} [{min(v)} to {max(v)}]" for k, v in vals.items())
            w1 = WEEK1[prec].get(c)
            print(f"  c={c:<3} {line}" + (f"  | week 1 tok/s {w1}" if w1 else ""))
        res[prec] = {"sweep": rows, "best": best, "repeats": spread}
    OUT["e3"] = res
    return res


# ── E0 ────────────────────────────────────────────────────
def e0():
    print("\n" + "=" * 70 + "\nE0  hardware ceiling vs measured (decode step = bytes / bandwidth)\n" + "=" * 70)
    res = {}
    for prec in ("bf16", "fp8"):
        sweep = runs(f"e3_{prec}_sweep")
        rows = [compare(prec, sweep[c][0]) for c in sorted(sweep)]
        res[prec] = rows
        print(f"\n{prec}: {'c':>3}{'pred step':>11}{'ITL p50':>9}{'eff':>6}{'TPOT p50':>10}{'eff':>6}{'ceiling tok/s':>15}{'meas':>8}")
        for r in rows:
            print(f"      {r['concurrency']:>3}{r['pred_step_ms']:>8} ms{r['meas_itl_p50_ms']:>6} ms{r['eff_vs_itl']:>6.0%}"
                  f"{r['meas_tpot_p50_ms']:>7} ms{r['eff_vs_tpot']:>6.0%}{r['pred_ceiling_tok_s']:>15}{r['meas_output_tok_s']:>8}")
    try:
        fit = fit_overhead(res["bf16"] + res["fp8"])
    except ZeroDivisionError:
        print("\n(c=1 refit skipped: bf16 and fp8 ITL identical)")
        OUT["e0"] = {"rows": res, "c1_fit": None}
        return res, None
    print(f"\nc=1 refit on week 2 data: effective bandwidth {fit['bandwidth_gb_s']} GB/s "
          f"({fit['bandwidth_gb_s'] / L4['bandwidth_gb_s']:.0%} of spec), fixed overhead {fit['overhead_ms']} ms per step")
    OUT["e0"] = {"rows": res, "c1_fit": fit}
    return res, fit


# ── E4 ────────────────────────────────────────────────────
def kernel_kind(name):
    n = name.lower()
    if any(k in n for k in ("flash", "attn", "attention", "paged", "fmha")):
        return "attention (KV)"
    if any(k in n for k in ("gemm", "gemv", "cutlass", "matmul", "sm89", "sm90", "scaled_mm", "fp8")):
        return "matmul (weights)"
    if "memcpy" in n or "memset" in n:
        return "memory copy"
    return "other (norms, activations, sampling)"


def union_len(intervals):
    total, end = 0.0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def e4(itl_by_c):
    print("\n" + "=" * 70 + "\nE4  where a bf16 decode step's time goes (torch profiler)\n" + "=" * 70)
    files = sorted(glob.glob(os.path.join(R, "e4_bf16", "traces", "**", "*.json*"), recursive=True))
    traces = []
    for p in files:
        try:
            opener = gzip.open if p.endswith(".gz") else open
            with opener(p, "rt") as f:
                t = json.load(f)
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {e}")
            continue
        ev = [e for e in t.get("traceEvents", []) if e.get("ph") == "X" and
              e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
        if ev:
            traces.append((min(e["ts"] for e in ev), p, ev))
    if not traces:
        print("  No GPU kernel events in the traces (with CUDA graphs, kernels can be hidden).")
        print("  Fall back to E0's two-precision fit for the overhead estimate.")
        OUT["e4"] = None
        return None
    traces.sort()
    labels = [1, 32] if len(traces) == 2 else list(range(1, len(traces) + 1))
    res = {}
    for c, (_, p, ev) in zip(labels, traces):
        start = min(e["ts"] for e in ev)
        end = max(e["ts"] + e["dur"] for e in ev)
        span = end - start
        busy = union_len([(e["ts"], e["ts"] + e["dur"]) for e in ev])
        by_kind = defaultdict(float)
        for e in ev:
            by_kind[kernel_kind(e.get("name", ""))] += e["dur"]
        itl = itl_by_c.get(c)
        steps = span / (itl * 1000) if itl else None
        res[c] = {
            "file": os.path.basename(p), "span_ms": round(span / 1000, 1),
            "gpu_busy_share": round(busy / span, 3),
            "per_step_ms": {"itl": itl,
                            "gpu_busy": round(busy / 1000 / steps, 1) if steps else None,
                            "gpu_idle": round((span - busy) / 1000 / steps, 1) if steps else None},
            "kernel_time_share": {k: round(v / sum(by_kind.values()), 3) for k, v in
                                  sorted(by_kind.items(), key=lambda kv: -kv[1])},
        }
        r = res[c]
        print(f"\nc={c}  ({r['file']}, {r['span_ms']} ms traced)")
        print(f"  GPU busy {r['gpu_busy_share']:.0%} of the time, idle {1 - r['gpu_busy_share']:.0%}")
        if steps:
            print(f"  per decode step (ITL {itl} ms): kernels {r['per_step_ms']['gpu_busy']} ms, "
                  f"gaps {r['per_step_ms']['gpu_idle']} ms")
        for k, v in r["kernel_time_share"].items():
            print(f"  {k:<38}{v:>6.0%} of kernel time")
    OUT["e4"] = res
    return res


# ── E2 ────────────────────────────────────────────────────
def e2():
    print("\n" + "=" * 70 + "\nE2  prefix caching on follow-up sessions (44 questions, 3 rounds)\n" + "=" * 70)
    res = {}
    for prec in ("bf16", "fp8"):
        for v in "ABC":
            files = sorted(glob.glob(os.path.join(R, "quality", f"quality_e2_{prec}_{v}_r*.json")))
            if not files:
                continue
            metas = [load(f)["meta"] for f in files]
            rows = load(files[0])["rows"]
            res[(prec, v)] = {
                "rounds": len(files),
                "hit_rate": st.median((m["prefix_cache_delta"] or {}).get("hit_rate", 0) for m in metas),
                "ttft_p50_ms": round(st.median(m["ttft_p50_s"] for m in metas) * 1000),
                "ttft_p95_ms": round(st.median(m["ttft_p95_s"] for m in metas) * 1000),
                "ttft_p50_range_ms": [round(min(m["ttft_p50_s"] for m in metas) * 1000),
                                      round(max(m["ttft_p50_s"] for m in metas) * 1000)],
                "prompt_tokens": round(st.mean(r["prompt_tokens"] for r in rows if r.get("prompt_tokens"))),
            }
    print(f"{'':<6}{'variant':<36}{'hit rate':>9}{'TTFT p50':>10}{'range':>12}{'p95':>7}{'vs A':>8}")
    names = {"A": "A: context first, score order", "B": "B: instructions first, score order",
             "C": "C: instructions first, document order"}
    for prec in ("bf16", "fp8"):
        base = res.get((prec, "A"))
        for v in "ABC":
            r = res.get((prec, v))
            if not r:
                continue
            vs = f"{r['ttft_p50_ms'] / base['ttft_p50_ms'] - 1:+.0%}" if base and v != "A" else ""
            rng = f"{r['ttft_p50_range_ms'][0]}-{r['ttft_p50_range_ms'][1]}"
            print(f"{prec:<6}{names[v]:<36}{r['hit_rate']:>9.1%}{r['ttft_p50_ms']:>8} ms{rng:>12}{r['ttft_p95_ms']:>7}{vs:>8}")
    print(f"\nprompt length {res[('bf16', 'A')]['prompt_tokens']} tokens on average; "
          "Phase 2 predicted 0.14 (score order) and 0.26 (document order) reusable from chunk IDs alone")
    OUT["e2"] = {f"{p}_{v}": r for (p, v), r in res.items()}
    return res


# ── E6 ────────────────────────────────────────────────────
def e6():
    print("\n" + "=" * 70 + "\nE6  FP8 at 32K context, 8,000-token prompts\n" + "=" * 70)
    log = os.path.join(R, "startup_fp8-32k.log")
    if os.path.exists(log):
        for line in open(log):
            if "KV cache size" in line or "Maximum concurrency" in line:
                print("  " + line.split("]")[-1].strip())
    long = runs("e6_fp8_32k")
    rows = [row(long[c][0]) for c in sorted(long)]
    print(f"{'c':>3}{'tok/s':>9}{'TTFT p50':>10}{'TTFT p95':>10}{'TPOT p50':>10}")
    for r in rows:
        print(f"{r['c']:>3}{r['out_tok_s']:>9}{r['ttft_p50']:>10}{r['ttft_p95']:>10}{r['tpot_p50']:>10}")
    OUT["e6"] = rows


# ── E7 ────────────────────────────────────────────────────
def counters(path):
    vals = {}
    if os.path.exists(path):
        for line in open(path):
            m = re.match(r"vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total\{.*\} ([\d.e+]+)", line)
            if m:
                vals[{"accepted_tokens": "accepted"}.get(m.group(1), m.group(1))] = float(m.group(2))
    return vals


def e7(e3res):
    print("\n" + "=" * 70 + "\nE7  FP8 + n-gram speculative decoding (prompt lookup, 5 tokens)\n" + "=" * 70)
    spec = runs("e7_fp8_spec")
    fp8 = {r["c"]: r for r in e3res["fp8"]["sweep"]}
    print(f"{'c':>3}{'FP8 tok/s':>11}{'+spec':>9}{'speedup':>9}{'TPOT p50':>10}{'+spec':>8}")
    rows = []
    for c in sorted(spec):
        s = row(spec[c][0])
        b = fp8.get(c)
        rows.append({"c": c, "fp8": b, "spec": s})
        if b:
            print(f"{c:>3}{b['out_tok_s']:>11}{s['out_tok_s']:>9}{s['out_tok_s'] / b['out_tok_s']:>8.2f}x"
                  f"{b['tpot_p50']:>10}{s['tpot_p50']:>8}")
    before = counters(os.path.join(R, "e7_fp8_spec", "spec_metrics_before.txt"))
    after = counters(os.path.join(R, "e7_fp8_spec", "spec_metrics_after.txt"))
    bench = {k: after.get(k, 0) - before.get(k, 0) for k in ("drafts", "draft_tokens", "accepted")}
    after_real = counters(os.path.join(R, "e7_fp8_spec", "spec_metrics_after_real.txt"))
    real = {k: after_real.get(k, 0) - after.get(k, 0) for k in bench}

    def acc(d):
        return {"acceptance": round(d["accepted"] / d["draft_tokens"], 3) if d["draft_tokens"] else None,
                "accepted_per_draft": round(d["accepted"] / d["drafts"], 2) if d["drafts"] else None, **d}
    bench, real = acc(bench), acc(real) if after_real else None
    print(f"\nacceptance on random-token benchmark: {bench['acceptance']:.0%} "
          f"({bench['accepted_per_draft']} extra tokens per draft)")
    if real:
        print(f"acceptance on 30 real RAG prompts:    {real['acceptance']:.0%} "
              f"({real['accepted_per_draft']} extra tokens per draft)")
    else:
        print("acceptance on real RAG prompts: no spec_metrics_after_real.txt, skipped")

    # decode speed on real prompts, one user: with spec (E7) vs without (E2 FP8 variant A, round 1)
    def decode_rate(path):
        rows_ = load(path)["rows"]
        rates = [r["output_tokens"] / (r["total_s"] - r["ttft_s"]) for r in rows_
                 if r.get("output_tokens") and r["output_tokens"] > 5 and r["total_s"] > r["ttft_s"]]
        return round(st.median(rates), 1) if rates else None
    with_spec = decode_rate(os.path.join(R, "quality", "quality_e7_fp8_spec_real_c1.json"))
    no_spec = decode_rate(os.path.join(R, "quality", "quality_e2_fp8_A_r1.json"))
    if with_spec and no_spec:
        print(f"decode speed on real RAG answers, one user: {no_spec} tok/s without, {with_spec} tok/s with "
              f"speculation ({with_spec / no_spec:.2f}x)  [different question sets: E2 sessions vs first 30 E1 questions]")
    OUT["e7"] = {"rows": rows, "bench": bench, "real": real,
                 "real_decode_tok_s": {"fp8": no_spec, "fp8_spec": with_spec}}


# ── charts ────────────────────────────────────────────────
def charts(e3res, e0rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(pip install matplotlib for charts)")
        return
    os.makedirs(os.path.join(R, "charts"), exist_ok=True)
    color = {"bf16": "#2a78d6", "fp8": "#eb6834"}      # categorical slots 1 and 2 (validated default palette)
    ink, muted, grid = "#2b2b29", "#6b6a64", "#e4e3dd"
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False})
    best = max(min(e3res[p]["best"]["c"] for p in ("bf16", "fp8") if e3res[p]["best"]), 1)

    def target_marker(ax, y_text, ymax=1.0):
        # last concurrency where both precisions meet the 2 s p95 TTFT target
        ax.axvline(best, ymax=ymax, color=muted, lw=1, ls=":")
        ax.text(best + 0.6, y_text, f"{best} users: last point\nwithin the latency target",
                color=muted, fontsize=9, va="top")

    def finish(ax, path, title, **legend):
        ax.set_title(title, loc="left", color=ink, fontsize=13, pad=12)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, labelcolor=ink, **legend)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"  saved {path}")

    print("\nCharts")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for prec in ("bf16", "fp8"):
        rows = e3res[prec]["sweep"]
        cs = [r["c"] for r in rows]
        ax.plot(cs, [r["out_tok_s"] for r in rows], color=color[prec], lw=2, marker="o", ms=6,
                label=f"{prec} measured")
        ax.plot(cs, [r["pred_ceiling_tok_s"] for r in e0rows[prec]], color=color[prec], lw=1.5, ls="--",
                label=f"{prec} decode ceiling (E0)")
    top = max(r["pred_ceiling_tok_s"] for r in e0rows["fp8"])
    target_marker(ax, top * 0.63, ymax=0.64)   # stop below the legend
    f16 = next((r for r in e3res["fp8"]["sweep"] if r["c"] == 16), None)
    if f16:
        ax.annotate("FP8 measured reaches\nbf16's theoretical ceiling", xy=(16, f16["out_tok_s"]),
                    xytext=(29, top * 0.06), color=ink, fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=muted, lw=1))
    ax.set_xlabel("concurrent users")
    ax.set_ylabel("output tokens / s")
    finish(ax, os.path.join(R, "charts", "throughput_vs_concurrency.png"),
           "Throughput vs concurrency on one L4, against the decode ceiling", loc="upper left")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for prec in ("bf16", "fp8"):
        rows = e3res[prec]["sweep"]
        ax.plot([r["c"] for r in rows], [r["ttft_p95"] / 1000 for r in rows], color=color[prec], lw=2,
                marker="o", ms=6, label=prec)
    ax.axhline(TTFT_TARGET_MS / 1000, color=muted, lw=1.2, ls="--", label="2 s target")
    target_marker(ax, max(r["ttft_p95"] for r in e3res["bf16"]["sweep"]) / 1000 * 0.97)
    ax.set_xlabel("concurrent users")
    ax.set_ylabel("p95 time to first token (s)")
    finish(ax, os.path.join(R, "charts", "p95_ttft_vs_concurrency.png"),
           "p95 time to first token vs concurrency", loc="lower right", bbox_to_anchor=(1, 0.22))


def main():
    e3res = e3()
    e0rows, _ = e0()
    itl = {}
    prof = runs("e4_bf16")
    for c, ds in prof.items():
        itl[c] = round(ds[0]["median_itl_ms"], 1)
    e4(itl)
    e2()
    e6()
    e7(e3res)
    charts(e3res, e0rows)
    with open(os.path.join(R, "analysis.json"), "w") as f:
        json.dump(OUT, f, indent=1, default=str)
    print(f"\nSaved {R}/analysis.json")


if __name__ == "__main__":
    main()