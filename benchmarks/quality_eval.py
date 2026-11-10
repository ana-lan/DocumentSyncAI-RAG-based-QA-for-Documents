"""
E1 and E2 runner: send frozen questions + contexts through a model, record the answers
and timings. Scoring is a separate step (benchmarks/score_eval.py), so a GPU run only
has to collect answers, and judging can happen later at no GPU cost.

Every request streams, so we record time to first token (TTFT) per question, the
number that prefix caching should move.

Usage (from repo root):
  # E1 on the 120-question set, against Groq for a local test
  python -m benchmarks.quality_eval --label groq_test --limit 5 --pause 3

  # E1 against vLLM in the cluster, 8 requests at a time
  python -m benchmarks.quality_eval --label e1_bf16 --base-url http://vllm:8000/v1 \
      --model Qwen/Qwen2.5-7B-Instruct --concurrency 8

  # E2 on the follow-up sessions: one user, in order, one variant per run
  python -m benchmarks.quality_eval --input eval/week2/frozen_sessions.json --label e2_bf16_C \
      --base-url http://vllm:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \
      --layout instructions_first --order position --reset-cache
"""
import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
from openai import OpenAI, RateLimitError, BadRequestError

from config import LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
from src.generation.generator import build_prompt, extract_citations, order_chunks

RESULTS_DIR = os.path.join("benchmarks", "results", "week2")
MAX_TOKENS = 256      # fixed cap so runs compare fairly; the answers here are short
_USAGE = {"stream_options": {"include_usage": True}}   # token counts; dropped if a server rejects it


# ── vLLM server helpers ───────────────────────────────────
def server_root(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


def scrape_prefix_metrics(base_url: str) -> dict:
    """
    Read vLLM's Prometheus /metrics page and keep every prefix-cache line.
    Values with several label sets are summed. Returns {} if the server has no
    /metrics page (Groq), so the runner works anywhere.
    """
    try:
        with urllib.request.urlopen(server_root(base_url) + "/metrics", timeout=5) as r:
            text = r.read().decode()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or "prefix_cache" not in line:
            continue
        name_part, _, value = line.rpartition(" ")
        name = name_part.split("{")[0]
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            pass
    return out


def cache_delta(before: dict, after: dict) -> dict:
    """Counter change during the run, plus hit rate = hit tokens / queried tokens if both exist."""
    delta = {k: after[k] - before.get(k, 0.0) for k in after}
    hits = next((v for k, v in delta.items() if "hits" in k), None)
    queries = next((v for k, v in delta.items() if "queries" in k), None)
    if hits is not None and queries:
        delta["hit_rate"] = round(hits / queries, 4)
    return delta


def reset_prefix_cache(base_url: str) -> bool:
    """
    Empty vLLM's prefix cache so one E2 variant can't benefit from another's run.
    The endpoint exists only when the server runs with VLLM_SERVER_DEV_MODE=1.
    """
    req = urllib.request.Request(server_root(base_url) + "/reset_prefix_cache", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception as e:
        print(f"WARNING: could not reset the prefix cache ({e}). Variants may share cached prefixes.")
        return False


# ── one question ──────────────────────────────────────────
def ask(client: OpenAI, model: str, row: dict, layout: str, order: str, pause: float) -> dict:
    retrieved = [{"text": t, "chunk_index": i, "score": s} for t, i, s in
                 zip(row["retrieved_texts"], row["retrieved_chunk_ids"], row["retrieved_scores"])]
    shown = order_chunks(retrieved, order)           # the order the model sees, [1]..[k]
    prompt = build_prompt(row["question"], retrieved, style="cite", layout=layout, order=order)

    for attempt in range(8):
        try:
            t0 = time.perf_counter()
            stream = client.chat.completions.create(
                model=model, temperature=0, max_tokens=MAX_TOKENS, stream=True,
                **_USAGE,
                messages=[{"role": "user", "content": prompt}],
            )
            parts, ttft, usage = [], None, None
            for event in stream:
                if event.usage:
                    usage = event.usage
                if event.choices and event.choices[0].delta.content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(event.choices[0].delta.content)
            total = time.perf_counter() - t0
            break
        except RateLimitError as e:
            wait = 20 * (attempt + 1)
            if attempt == 0:
                print(f"    rate limit: {str(e)[:200]}")   # says which limit: per minute or per day
            print(f"    waiting {wait}s")
            time.sleep(wait)
        except BadRequestError:
            if not _USAGE:
                raise
            print("    server rejected stream_options; continuing without token counts")
            _USAGE.clear()
    else:
        raise RuntimeError(f"{row['id']}: failed after retries")

    answer = "".join(parts).strip()
    if pause:
        time.sleep(pause)
    keep = ("id", "paper", "type", "question", "answer_kind", "evidence", "base_id",
            "wrong_answer", "session", "turn", "hit", "hit_rank")
    return {
        **{k: row[k] for k in keep if k in row},
        "gold_answer": row["answer"],
        "model_answer": answer,
        "citations": extract_citations(answer),
        "shown_chunk_ids": [c["chunk_index"] for c in shown],
        "shown_texts": [c["text"] for c in shown],
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "ttft_s": round(ttft if ttft is not None else total, 4),
        "total_s": round(total, 4),
    }


def pct(values: list[float], q: int) -> float:
    return round(float(np.percentile(values, q)), 4) if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="eval/week2/frozen_eval.json")
    ap.add_argument("--label", required=True)
    ap.add_argument("--base-url", default=LLM_BASE_URL)
    ap.add_argument("--model", default=LLM_MODEL)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--layout", choices=["context_first", "instructions_first"], default="context_first")
    ap.add_argument("--order", choices=["score", "position"], default="score")
    ap.add_argument("--reset-cache", action="store_true", help="empty vLLM's prefix cache before the run")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pause", type=float, default=0.0, help="seconds between requests (Groq rate limits)")
    args = ap.parse_args()

    with open(args.input) as f:
        rows = json.load(f)["rows"][: args.limit]
    sessions = "session" in rows[0]
    if sessions and args.concurrency != 1:
        print("Sessions run one question at a time, in order, like one user. Using concurrency 1.")
        args.concurrency = 1

    api_key = LLM_API_KEY if "groq" in args.base_url else "EMPTY"
    client = OpenAI(base_url=args.base_url, api_key=api_key, timeout=120, max_retries=0)

    reset_ok = reset_prefix_cache(args.base_url) if args.reset_cache else None
    before = scrape_prefix_metrics(args.base_url)
    print(f"{len(rows)} questions | {args.model} @ {args.base_url} | layout={args.layout} "
          f"order={args.order} concurrency={args.concurrency}")

    started = time.perf_counter()
    failed, errors = None, []
    if args.concurrency == 1:
        out = []
        for i, row in enumerate(rows, 1):
            try:
                out.append(ask(client, args.model, row, args.layout, args.order, args.pause))
            except Exception as e:          # keep the answers collected so far
                failed = f"{row['id']}: {e}"
                print(f"STOPPED at question {i}: {failed}")
                break
            print(f"[{i:03d}/{len(rows)}] ttft {out[-1]['ttft_s']:.3f}s | {out[-1]['model_answer'][:70]!r}")
        else:
            def safe_ask(r):      # one failed question must not throw away the others' answers
                try:
                    return ask(client, args.model, r, args.layout, args.order, args.pause), None
                except Exception as e:
                    return None, f"{r['id']}: {e}"
            with ThreadPoolExecutor(args.concurrency) as pool:
                results = list(pool.map(safe_ask, rows))
            out = [res for res, _ in results if res is not None]
            errors = [err for _, err in results if err]
            if errors:
                failed = f"{len(errors)} of {len(rows)} questions failed, first: {errors[0]}"
                print(f"FAILED {len(errors)}/{len(rows)} questions; saving the {len(out)} answers that worked")
    wall = time.perf_counter() - started
    after = scrape_prefix_metrics(args.base_url)

    ttfts = [r["ttft_s"] for r in out]
    meta = {
        "label": args.label, "input": args.input, "model": args.model, "base_url": args.base_url,
        "layout": args.layout, "order": args.order, "temperature": 0, "max_tokens": MAX_TOKENS,
        "concurrency": args.concurrency, "n": len(out), "wall_s": round(wall, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ttft_p50_s": pct(ttfts, 50), "ttft_p95_s": pct(ttfts, 95),
        "complete": failed is None, "stopped_at": failed, "errors": errors,
        "prefix_cache_reset": reset_ok,
        "prefix_cache_delta": cache_delta(before, after) if after else None,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"quality_{args.label}.json")
    with open(path, "w") as f:
        json.dump({"meta": meta, "rows": out}, f, indent=1)

    print(f"\nTTFT p50 {meta['ttft_p50_s']}s  p95 {meta['ttft_p95_s']}s | wall {meta['wall_s']}s")
    if meta["prefix_cache_delta"]:
        print(f"prefix cache during run: {meta['prefix_cache_delta']}")
    print(f"Saved to {path}")
    if failed:
        raise SystemExit(f"Partial run saved ({len(out)}/{len(rows)} answers). Not a complete result.")


if __name__ == "__main__":
    main()