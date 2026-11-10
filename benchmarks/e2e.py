# benchmarks/e2e.py
"""
End-to-end RAG benchmark for DocumentSync.

Ingests one PDF into an in-memory ChromaDB collection, then runs a fixed set
of questions sequentially (one user) through retrieve + generate, timing each
stage. Week 2 adds quality metrics next to the timings:

  hit@k     share of questions where a retrieved chunk contains the gold evidence
  MRR       mean of 1/rank of that chunk (1.0 = always ranked first)
  cite      share of answers whose cited passage really contains the evidence
  correct   first-pass answer check (misses are listed for a human look)

Usage (from repo root):
  python -m benchmarks.e2e --label groq_local
  python -m benchmarks.e2e --label tol3 --retrieval-only --x-tolerance 3
  python -m benchmarks.e2e --label before --x-tolerance 3 --keep-references --prompt-style plain
  LLM_BASE_URL=http://vllm:8000/v1 LLM_MODEL=Qwen/Qwen2.5-7B-Instruct python -m benchmarks.e2e --label vllm_l4
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (EMBEDDING_MODEL, LLM_BASE_URL, LLM_MODEL, TEMPERATURE,
                    PDF_X_TOLERANCE, DROP_REFERENCES, PROMPT_STYLE, DEFAULT_TOP_K)
from src.ingestion.parser import parse_pdf
from src.ingestion.cleaner import clean_pages
from src.ingestion.chunker import chunk_fixed
from src.retrieval.retriever import retrieve
from src.generation.generator import generate
from benchmarks.metrics import (contains_any, hit_rank, reciprocal_rank, is_refusal,
                                answer_matches, citation_correct)

HERE = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(HERE, "attention.pdf")
RESULTS_DIR = os.path.join(HERE, "results")
COLLECTION = "e2e_bench"

WARMUP_QUESTIONS = [
    "What is the Transformer?",
    "What does the encoder do?",
]

# The 20 week 1 questions. For each one:
#   expected: the answer from the paper (for humans reading the results)
#   evidence: phrases that only the passage holding the answer contains. A retrieved
#             chunk "hits" if it contains any of them (whitespace-insensitive).
#   key:      first-pass answer check, see benchmarks/metrics.answer_matches
QUESTIONS = [
    {"q": "What BLEU score does the Transformer achieve on WMT 2014 English-to-German?",
     "expected": "28.4", "evidence": ["28.4 BLEU", "BLEU score of 28.4"], "key": [["28.4"]]},
    {"q": "What BLEU score does the big Transformer achieve on WMT 2014 English-to-French?",
     "expected": "41.8", "evidence": ["41.8"], "key": [["41.8"]]},
    {"q": "How many identical layers are in the encoder stack?",
     "expected": "6", "evidence": ["N = 6 identical layers"], "key": [["6", "six"]]},
    {"q": "What is the dimensionality of the model outputs, d_model?",
     "expected": "512", "evidence": ["produce outputs of dimension d = 512", "input and output is d = 512"], "key": [["512"]]},
    {"q": "How many attention heads does the base model use?",
     "expected": "8", "evidence": ["h = 8 parallel attention layers"], "key": [["8", "eight"]]},
    {"q": "What is the inner-layer dimensionality of the feed-forward network?",
     "expected": "2048", "evidence": ["inner-layer has dimensionality"], "key": [["2048"]]},
    {"q": "Why are the dot products scaled by the square root of d_k?",
     "expected": "large values push softmax into small-gradient regions",
     "evidence": ["extremely small gradients"], "key": [["gradient"]]},
    {"q": "Which optimizer was used for training?",
     "expected": "Adam (beta1=0.9, beta2=0.98, eps=1e-9)", "evidence": ["Adam optimizer"], "key": [["adam"]]},
    {"q": "How many warmup steps does the learning rate schedule use?",
     "expected": "4000", "evidence": ["warmup_steps = 4000", "warmup steps = 4000"], "key": [["4000", "4,000"]]},
    {"q": "What value of label smoothing was used?",
     "expected": "0.1", "evidence": ["label smoothing of value"], "key": [["0.1"]]},
    {"q": "What hardware was used to train the models?",
     "expected": "8 NVIDIA P100 GPUs", "evidence": ["8 NVIDIA P100"], "key": [["p100"]]},
    {"q": "How long did the big model take to train?",
     "expected": "3.5 days / 300,000 steps", "evidence": ["3.5 days"], "key": [["3.5"]]},
    {"q": "What dropout rate was used for the base model?",
     "expected": "0.1", "evidence": ["we use a rate of"], "key": [["0.1"]]},
    {"q": "What functions are used for the positional encodings?",
     "expected": "sine and cosine", "evidence": ["sine and cosine functions"],
     "key": [["sine and cosine", "sin and cos", "sinusoid"]]},
    {"q": "Why did the authors choose self-attention over recurrent layers?",
     "expected": "complexity, parallelization, path length",
     "evidence": ["total computational complexity per layer"], "key": [["complexity", "parallel", "path length"]]},
    {"q": "Why is masking used in the decoder self-attention?",
     "expected": "prevent attending to subsequent positions",
     "evidence": ["prevent positions from attending to subsequent positions"],
     "key": [["subsequent", "future", "later position"]]},
    {"q": "Besides translation, what task was the Transformer evaluated on?",
     "expected": "English constituency parsing", "evidence": ["English constituency parsing"],
     "key": [["constituency"]]},
    {"q": "What vocabulary size was used for English-German byte-pair encoding?",
     "expected": "about 37000 tokens", "evidence": ["37000 tokens"], "key": [["37000", "37,000", "37k"]]},
    {"q": "What is the per-layer complexity of self-attention?",
     "expected": "O(n^2 * d)", "evidence": ["O(n2 · d)", "O(n2d)", "O(n^2 · d)"],
     "key": [["n2", "n^2", "n^{2}", "n²"]]},
    {"q": "What beam size was used during inference?",
     "expected": "4, with length penalty 0.6", "evidence": ["beam size of 4"], "key": [["4", "four"]]},
]


def ingest(client, x_tolerance: float, drop_references: bool) -> list[str]:
    """Parse, clean, chunk (fixed_1024, same as the app), embed and store the PDF. Returns the chunks."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    pages = parse_pdf(PDF_PATH, x_tolerance=x_tolerance)
    chunks = chunk_fixed(text=clean_pages(pages, drop_references=drop_references),
                         strategy_name="fixed_1024")
    embeddings = model.encode(chunks)

    collection = client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=[{"chunk_index": i} for i in range(len(chunks))],
    )
    return chunks


def run_one(item: dict, client, retrieval_only: bool, prompt_style: str) -> dict:
    """Run one question through the real pipeline, timing each stage and scoring it."""
    t0 = time.perf_counter()
    chunks = retrieve(query=item["q"], strategy_name=COLLECTION, client=client)
    t1 = time.perf_counter()
    answer, citations = "", []
    if not retrieval_only:
        out = generate(query=item["q"], retrieved_chunks=chunks, prompt_style=prompt_style)
        answer, citations = out["answer"], out["citations"]
    t2 = time.perf_counter()

    texts = [c["text"] for c in chunks]
    rank = hit_rank(texts, item["evidence"])
    row = {
        "question": item["q"],
        "expected": item["expected"],
        "answer": answer,
        "num_chunks": len(chunks),
        "retrieved_chunk_ids": [c["chunk_index"] for c in chunks],
        "retrieved_scores": [round(c["score"], 4) for c in chunks],
        "hit_rank": rank,
        "retrieve_s": t1 - t0,
        "generate_s": t2 - t1,
        "total_s": t2 - t0,
    }
    if not retrieval_only:
        row["citations"] = citations
        row["refused"] = is_refusal(answer)
        row["answer_ok"] = answer_matches(answer, item["key"])
        row["citation_ok"] = citation_correct(citations, texts, item["evidence"])
    return row


def stats(rows: list[dict], key: str) -> dict:
    values = [r[key] for r in rows]
    return {
        "p50": round(float(np.percentile(values, 50)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
        "mean": round(float(np.mean(values)), 3),
    }


def quality(rows: list[dict], retrieval_only: bool) -> dict:
    n = len(rows)
    q = {
        f"hit@{DEFAULT_TOP_K}": round(sum(r["hit_rank"] is not None for r in rows) / n, 3),
        "mrr": round(sum(reciprocal_rank(r["hit_rank"]) for r in rows) / n, 3),
        "evidence_missing_from_doc": sum(r["evidence_in_doc"] is False for r in rows),
    }
    if not retrieval_only:
        cited = [r for r in rows if r["citation_ok"] is not None]
        answered = [r for r in rows if not r["refused"]]
        q.update({
            "correct": sum(r["answer_ok"] for r in rows),
            "refused": sum(r["refused"] for r in rows),
            "citation_rate": round(len([r for r in answered if r["citations"]]) / max(len(answered), 1), 3),
            "citation_correct": round(sum(r["citation_ok"] for r in cited) / max(len(cited), 1), 3),
        })
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="name for this run, e.g. groq_local or vllm_l4")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM call, time and score retrieval only")
    parser.add_argument("--limit", type=int, default=len(QUESTIONS), help="run only the first N questions")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to sleep between questions (not timed)")
    parser.add_argument("--x-tolerance", type=float, default=PDF_X_TOLERANCE, help="pdfplumber x_tolerance")
    parser.add_argument("--keep-references", action="store_true", help="do not cut the references section")
    parser.add_argument("--prompt-style", choices=["plain", "cite"], default=PROMPT_STYLE)
    args = parser.parse_args()
    drop_refs = DROP_REFERENCES and not args.keep_references

    client = chromadb.EphemeralClient()
    all_chunks = ingest(client, args.x_tolerance, drop_refs)
    print(f"Backend: {LLM_BASE_URL} | Model: {LLM_MODEL} | Chunks indexed: {len(all_chunks)}")
    print(f"x_tolerance={args.x_tolerance} drop_references={drop_refs} "
          f"prompt={args.prompt_style} temperature={TEMPERATURE}")

    if not args.retrieval_only:
        print(f"Warming up with {len(WARMUP_QUESTIONS)} questions (not counted)...")
        for q in WARMUP_QUESTIONS:
            run_one({"q": q, "expected": "", "evidence": [""], "key": [[""]]},
                    client, args.retrieval_only, args.prompt_style)
            time.sleep(args.pause)

    rows = []
    questions = QUESTIONS[: args.limit]
    for i, item in enumerate(questions, start=1):
        row = run_one(item, client, args.retrieval_only, args.prompt_style)
        # Is the evidence anywhere in the document? If not, either extraction mangled
        # it or the evidence phrase is wrong; either way the question can't hit.
        row["evidence_in_doc"] = any(contains_any(c, item["evidence"]) for c in all_chunks)
        rows.append(row)
        rank = row["hit_rank"] or "-"
        status = "" if args.retrieval_only else (" ok " if row["answer_ok"] else " MISS")
        print(f"[{i:02d}/{len(questions)}]{status} rank {rank} | total {row['total_s']:.2f}s "
              f"| {row['answer'][:60]!r}")
        time.sleep(args.pause)

    summary = {
        "label": args.label,
        "backend": LLM_BASE_URL,
        "model": LLM_MODEL,
        "retrieval_only": args.retrieval_only,
        "x_tolerance": args.x_tolerance,
        "drop_references": drop_refs,
        "prompt_style": args.prompt_style,
        "temperature": TEMPERATURE,
        "num_chunks_indexed": len(all_chunks),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": len(rows),
        "quality": quality(rows, args.retrieval_only),
        "retrieve_s": stats(rows, "retrieve_s"),
        "generate_s": stats(rows, "generate_s"),
        "total_s": stats(rows, "total_s"),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"e2e_{args.label}.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    print("\n=== Quality ===")
    for k, v in summary["quality"].items():
        print(f"{k:<26} {v}")
    missing = [r["question"] for r in rows if not r["evidence_in_doc"]]
    if missing:
        print("\nEvidence not found anywhere in the document (check extraction or the phrase):")
        for q in missing:
            print(f"  - {q}")
    if not args.retrieval_only:
        misses = [r for r in rows if not r["answer_ok"]]
        if misses:
            print("\nAnswers flagged by the first-pass check (read these by eye):")
            for r in misses:
                print(f"  - {r['question']}\n      expected {r['expected']!r} | got {r['answer'][:100]!r}")
    print("\n=== Latency (seconds) ===")
    for stage in ("retrieve_s", "generate_s", "total_s"):
        s = summary[stage]
        print(f"{stage:<11} p50 {s['p50']:.3f}  p95 {s['p95']:.3f}  mean {s['mean']:.3f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
