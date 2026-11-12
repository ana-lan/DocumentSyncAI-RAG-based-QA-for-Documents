"""
Export E1 scored answers to JSONL for memory-agreement-check's compare mode
(https://github.com/vadimchernets/memory-agreement-check).

Usage (from repo root):
  python -m benchmarks.export_jsonl
"""
import json

Q = "benchmarks/results/gke_week2/quality/"

for name in ("bf16", "fp8"):
    rows = json.load(open(f"{Q}scored_e1_{name}.json"))["rows"]
    out = f"{Q}e1_{name}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps({
                "qid": r["id"],
                "answer": r["model_answer"],
                "correct": bool(r["correct"]),
                "category": r["type"],
            }) + "\n")
    print(f"{out}: {len(rows)} rows")