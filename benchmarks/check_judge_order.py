"""
Review check: the judge cache key uses the SET of passages (sorted), not their order.
In E2, A and C show the same passages in different orders, so an answer that is word for
word the same in A and C reused A's verdict for C. This re-judges exactly those answers
with C's real passage order, with no cache, and reports whether any verdict changes.

Read-only: it never writes eval/week2/judge_cache.json.

Usage (from repo root, needs a Groq key with some daily quota left):
  python -m benchmarks.check_judge_order
"""
import json

import benchmarks.score_eval as se

se.save_cache = lambda cache: None          # never touch judge_cache.json from this script
Q = "benchmarks/results/gke_week2/quality/"


def main():
    a = {r["id"]: r for r in json.load(open(Q + "scored_e2_bf16_A_r1.json"))["rows"]}
    c = {r["id"]: r for r in json.load(open(Q + "scored_e2_bf16_C_r1.json"))["rows"]}
    cache = se.load_cache()
    shared = [i for i in sorted(c) if i in a and not c[i]["refused"]
              and c[i]["model_answer"] == a[i]["model_answer"]]
    print(f"{len(shared)} non-refused answers identical in A and C (these reused A's verdict)")

    changed = 0
    for i in shared:
        row = c[i]
        reordered = row["shown_texts"] != a[i]["shown_texts"]
        old = se.judge(row, cache)          # cached verdict, no API call
        new = se.judge(row, {})             # fresh verdict with C's passage order
        diff = old != new
        changed += diff
        print(f"  {i:<22} order differs: {reordered!s:<5}  cached {old}  fresh {new}  {'CHANGED' if diff else 'same'}")
    print(f"\n{changed} of {len(shared)} verdicts changed with the real passage order")


if __name__ == "__main__":
    main()