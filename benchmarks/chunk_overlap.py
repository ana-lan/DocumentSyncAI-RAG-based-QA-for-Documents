"""
E2 step 0: how much could prefix caching ever help real RAG prompts?

vLLM's prefix cache reuses work only for the START of a prompt that exactly
matches the start of an earlier prompt. Our prompt is:

    Context: [1] chunk [2] chunk ... [5] chunk   Question: ...

So a new question saves work only if its first retrieved chunks are the same
chunks, in the same order, as an earlier question's. This script measures that
from the chunk IDs a benchmark run already saved (no GPU, no LLM).

For each question, against all earlier questions in the same session:
  seen      share of its 5 chunks that some earlier question also retrieved
            (the absolute ceiling, if order didn't matter)
  prefix    leading chunks identical to an earlier prompt, chunks in score order (today's prompt)
  prefix_pos  the same, with chunks sorted by position in the document (variant C)

Chunk IDs are the chunk's position in the document, so sorting the IDs gives position order.

Usage (from repo root):
  python -m benchmarks.chunk_overlap benchmarks/results/e2e_e5_after.json
  python -m benchmarks.chunk_overlap <file> --session-key session     (Phase 3 follow-up sessions)
"""
import argparse
import json
from collections import defaultdict
from itertools import combinations


def common_prefix(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def session_overlap(sessions: dict[str, list[list[int]]]) -> dict:
    seen, prefix, prefix_pos, jaccard = [], [], [], []
    for ids_list in sessions.values():
        for a, b in combinations(ids_list, 2):
            if a and b:
                jaccard.append(len(set(a) & set(b)) / len(set(a) | set(b)))
        for i in range(1, len(ids_list)):
            cur, earlier = ids_list[i], ids_list[:i]
            if not cur:
                continue
            k = len(cur)
            earlier_ids = set().union(*map(set, earlier))
            seen.append(len(set(cur) & earlier_ids) / k)
            prefix.append(max(common_prefix(cur, e) for e in earlier) / k)
            prefix_pos.append(max(common_prefix(sorted(cur), sorted(e)) for e in earlier) / k)

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    return {
        "questions_scored": len(seen),
        "pairwise_jaccard": mean(jaccard),
        "seen_before": mean(seen),
        "prefix_reuse_score_order": mean(prefix),
        "prefix_reuse_position_order": mean(prefix_pos),
        "share_with_any_prefix_hit_score": mean([p > 0 for p in prefix]),
        "share_with_any_prefix_hit_position": mean([p > 0 for p in prefix_pos]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="a results JSON whose rows have retrieved_chunk_ids")
    parser.add_argument("--session-key", default=None,
                        help="row field that groups questions into sessions; default: the whole file is one session")
    args = parser.parse_args()

    with open(args.results) as f:
        rows = json.load(f)["rows"]

    sessions = defaultdict(list)
    for r in rows:
        key = r.get(args.session_key, "all") if args.session_key else "all"
        sessions[key].append(r["retrieved_chunk_ids"])

    result = session_overlap(sessions)
    print(f"{len(rows)} questions in {len(sessions)} session(s)\n")
    for k, v in result.items():
        print(f"{k:<38} {v}")

    print("\nHow to read it: prefix_reuse is roughly the share of each prompt's context that the")
    print("prefix cache could skip. Under about 0.2 (less than one chunk in five), E2's prompt")
    print("variants can't show a real effect, and the low overlap is itself the finding.")


if __name__ == "__main__":
    main()