# benchmarks/e2e.py
"""
End-to-end RAG latency benchmark for DocumentSync.

Ingests one PDF into an in-memory ChromaDB collection, then runs a fixed
set of questions sequentially (one user) through retrieve + generate,
timing each stage. Writes per-question rows and a p50/p95 summary to JSON.

Usage (from repo root):
  python -m benchmarks.e2e --label groq_local
  LLM_BASE_URL=http://vllm:8000/v1 LLM_MODEL=Qwen/Qwen2.5-7B-Instruct python -m benchmarks.e2e --label vllm_l4
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import chromadb
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, LLM_BASE_URL, LLM_MODEL
from src.ingestion.cleaner import clean_pages
from src.ingestion.chunker import chunk_fixed
from src.retrieval.retriever import retrieve
from src.generation.generator import generate

HERE = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(HERE, "attention.pdf")
RESULTS_DIR = os.path.join(HERE, "results")
COLLECTION = "e2e_bench"

WARMUP_QUESTIONS = [
    "What is the Transformer?",
    "What does the encoder do?",
]

# (question, expected answer from the paper) for a quick quality spot-check
QUESTIONS = [
    ("What BLEU score does the Transformer achieve on WMT 2014 English-to-German?", "28.4"),
    ("What BLEU score does the big Transformer achieve on WMT 2014 English-to-French?", "41.8"),
    ("How many identical layers are in the encoder stack?", "6"),
    ("What is the dimensionality of the model outputs, d_model?", "512"),
    ("How many attention heads does the base model use?", "8"),
    ("What is the inner-layer dimensionality of the feed-forward network?", "2048"),
    ("Why are the dot products scaled by the square root of d_k?", "large values push softmax into small-gradient regions"),
    ("Which optimizer was used for training?", "Adam (beta1=0.9, beta2=0.98, eps=1e-9)"),
    ("How many warmup steps does the learning rate schedule use?", "4000"),
    ("What value of label smoothing was used?", "0.1"),
    ("What hardware was used to train the models?", "8 NVIDIA P100 GPUs"),
    ("How long did the big model take to train?", "3.5 days / 300,000 steps"),
    ("What dropout rate was used for the base model?", "0.1"),
    ("What functions are used for the positional encodings?", "sine and cosine"),
    ("Why did the authors choose self-attention over recurrent layers?", "complexity, parallelization, path length"),
    ("Why is masking used in the decoder self-attention?", "prevent attending to subsequent positions"),
    ("Besides translation, what task was the Transformer evaluated on?", "English constituency parsing"),
    ("What vocabulary size was used for English-German byte-pair encoding?", "about 37000 tokens"),
    ("What is the per-layer complexity of self-attention?", "O(n^2 * d)"),
    ("What beam size was used during inference?", "4, with length penalty 0.6"),
]


def ingest(client) -> int:
    """Parse, clean, chunk (fixed_1024, same as the app), embed and store the PDF."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    pages = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text)

    chunks = chunk_fixed(text=clean_pages(pages), strategy_name="fixed_1024")
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
    return len(chunks)


def run_one(question: str, client, retrieval_only: bool = False) -> dict:
    """Run one question through the real pipeline, timing each stage."""
    t0 = time.perf_counter()
    chunks = retrieve(query=question, strategy_name=COLLECTION, client=client)
    t1 = time.perf_counter()
    if retrieval_only:
        answer = ""
    else:
        answer = generate(query=question, retrieved_chunks=chunks)["answer"]
    t2 = time.perf_counter()
    return {
        "question": question,
        "answer": answer,
        "num_chunks": len(chunks),
        "retrieve_s": t1 - t0,
        "generate_s": t2 - t1,
        "total_s": t2 - t0,
    }


def stats(rows: list[dict], key: str) -> dict:
    values = [r[key] for r in rows]
    return {
        "p50": round(float(np.percentile(values, 50)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
        "mean": round(float(np.mean(values)), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="name for this run, e.g. groq_local or vllm_l4")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM call, time retrieval only")
    parser.add_argument("--limit", type=int, default=len(QUESTIONS), help="run only the first N questions")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to sleep between questions (not timed)")
    args = parser.parse_args()

    client = chromadb.EphemeralClient()
    n_chunks = ingest(client)
    print(f"Backend: {LLM_BASE_URL} | Model: {LLM_MODEL} | Chunks indexed: {n_chunks}")

    print(f"Warming up with {len(WARMUP_QUESTIONS)} questions (not counted)...")
    for q in WARMUP_QUESTIONS:
        run_one(q, client, args.retrieval_only)
        time.sleep(args.pause)

    rows = []
    questions = QUESTIONS[: args.limit]
    for i, (question, expected) in enumerate(questions, start=1):
        row = run_one(question, client, args.retrieval_only)
        row["expected"] = expected
        rows.append(row)
        print(f"[{i:02d}/{len(questions)}] total {row['total_s']:.2f}s "
              f"(retrieve {row['retrieve_s']:.3f}s, generate {row['generate_s']:.2f}s) "
              f"| {row['answer'][:60]!r}")
        time.sleep(args.pause)

    summary = {
        "label": args.label,
        "backend": LLM_BASE_URL,
        "model": LLM_MODEL,
        "retrieval_only": args.retrieval_only,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": len(rows),
        "retrieve_s": stats(rows, "retrieve_s"),
        "generate_s": stats(rows, "generate_s"),
        "total_s": stats(rows, "total_s"),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"e2e_{args.label}.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    print("\n=== Summary (seconds) ===")
    for stage in ("retrieve_s", "generate_s", "total_s"):
        s = summary[stage]
        print(f"{stage:<11} p50 {s['p50']:.3f}  p95 {s['p95']:.3f}  mean {s['mean']:.3f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()