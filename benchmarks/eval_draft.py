"""
E1, Phase 3 part A: draft the stratified eval set with an LLM, keeping only
questions whose evidence really exists in the paper.

Steps (run from repo root):
  python -m benchmarks.eval_draft chunks                 # chunk the 3 papers exactly like the app does
  python -m benchmarks.eval_draft draft --paper attention
  python -m benchmarks.eval_draft draft --paper rag
  python -m benchmarks.eval_draft draft --paper lora
  python -m benchmarks.eval_draft csv                    # one review sheet for all papers

Each question gets a type, a gold answer and gold evidence: a short exact quote
from the paper. A drafted question is dropped automatically if its quote can't
be found in the paper's chunks, so every kept question is traceable to the text.
A human still reviews every kept question before it enters the set.
"""
import argparse
import csv
import json
import os
import re
import time

from openai import OpenAI, RateLimitError

from config import LLM_BASE_URL, LLM_MODEL
from src.ingestion.parser import parse_pdf
from src.ingestion.cleaner import clean_pages
from src.ingestion.chunker import chunk_fixed
from benchmarks.metrics import squash

OUT_DIR = "eval/week2"
CHUNK_DIR = os.path.join(OUT_DIR, "chunks")
PAPERS = {
    "attention": {"pdf": "benchmarks/attention.pdf", "arxiv": "1706.03762",
                  "title": "Attention Is All You Need"},
    "rag": {"pdf": "data/raw/rag.pdf", "arxiv": "2005.11401",
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"},
    "lora": {"pdf": "data/raw/lora.pdf", "arxiv": "2106.09685",
             "title": "LoRA: Low-Rank Adaptation of Large Language Models"},
}
TYPES = ["direct", "paraphrased", "verbose", "keyword", "plain_language",
         "multi_hop", "near_identical", "unanswerable"]
BATCH = 4          # passages shown to the LLM per fact-drafting call
N_EXTRA = 8        # multi-hop and unanswerable drafts per paper
DRAFT_MODEL = os.getenv("DRAFT_MODEL", LLM_MODEL)


# ── LLM helper ────────────────────────────────────────────
# Both Groq keys are used in turn, which roughly doubles the rate limit.
_KEYS = [k for k in (os.getenv("GROQ_API_KEY"), os.getenv("GROQ_JUDGE_API_KEY")) if k]
_clients = [OpenAI(base_url=LLM_BASE_URL, api_key=k) for k in _KEYS]
_turn = 0


def ask_json(prompt: str) -> dict:
    """Send a prompt, return the JSON object in the reply. Retries through rate limits."""
    global _turn
    for attempt in range(10):
        client = _clients[_turn % len(_clients)]
        _turn += 1
        try:
            reply = client.chat.completions.create(
                model=DRAFT_MODEL, temperature=0.3, max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            ).choices[0].message.content
        except RateLimitError:
            wait = 20 * (attempt + 1)
            print(f"    rate limit, waiting {wait}s")
            time.sleep(wait)
            continue
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)   # some models think out loud first
        start, end = reply.find("{"), reply.rfind("}")
        try:
            return json.loads(reply[start:end + 1])
        except (ValueError, json.JSONDecodeError):
            print("    reply was not valid JSON, retrying")
    raise RuntimeError("LLM call failed 10 times")


# ── Chunks ────────────────────────────────────────────────
def build_chunks():
    """Chunk every paper with the app's pipeline and save it, so IDs stay fixed from here on."""
    os.makedirs(CHUNK_DIR, exist_ok=True)
    for name, p in PAPERS.items():
        chunks = chunk_fixed(clean_pages(parse_pdf(p["pdf"])), "fixed_1024")
        with open(os.path.join(CHUNK_DIR, f"{name}.json"), "w") as f:
            json.dump({"paper": name, "arxiv": p["arxiv"], "chunks": chunks}, f, indent=1)
        print(f"{name}: {len(chunks)} chunks")


def load_chunks(name: str) -> list[str]:
    with open(os.path.join(CHUNK_DIR, f"{name}.json")) as f:
        return json.load(f)["chunks"]


def find_evidence(chunks: list[str], quote: str) -> list[int]:
    """IDs of every chunk containing the quote (whitespace-insensitive). Empty = quote not in paper."""
    q = squash(quote or "")
    if len(q) < 10:
        return []
    return [i for i, c in enumerate(chunks) if q in squash(c)]


# ── Prompts ───────────────────────────────────────────────
FACT_PROMPT = """You are building a test set for a question-answering system over the research paper "{title}".
Below are consecutive passages from the paper.

{passages}

Return a JSON object with two keys.

"fact": one question a reader might ask whose answer is stated explicitly in ONE passage above.
Prefer specific facts: numbers, names, settings, results. Fields:
  "question", "answer" (short),
  "evidence": an exact quote of 5 to 15 words copied character for character from the passage;
              if the answer is a number, the quote must contain it.

"confusable": if the passages report similar values for two different settings (base vs large
model, two datasets, two methods), one question about ONE of them that someone could easily
answer with the other's value. The question must name its exact setting. Fields:
  "question", "answer", "wrong_answer" (the other setting's value), "evidence" (exact quote containing the answer).
Use null when there is no such pair.

Return only the JSON object."""

VARIANT_PROMPT = """Rewrite this question about the paper "{title}" in four styles. Every rewrite must have exactly the same answer.

Question: {question}
Answer: {answer}

Return a JSON object with these keys:
"paraphrased": an indirect version that avoids the question's key terms where possible
               (for example "How well did it do on translation?" instead of "What BLEU score...")
"verbose": two or three sentences about why the reader is asking, then the question
"keyword": a terse search-box query of 2 to 5 words, no question mark
"plain_language": the same question for someone with no machine learning background, no jargon
Return only the JSON object."""

MULTIHOP_PROMPT = """Here are facts from the paper "{title}", each with an id:

{facts}

Write {n} questions that each need TWO of these facts to answer, such as a comparison, a difference,
or a combination ("How many more X than Y...", "Which is larger..."). Return a JSON object:
{{"questions": [{{"question": "...", "answer": "...", "uses": [id1, id2]}}]}}"""

UNANSWERABLE_PROMPT = """The paper "{title}" covers questions like these:

{questions}

Write {n} questions a reader might plausibly ask about this paper that the paper does NOT answer:
details it never reports (cost, energy, a dataset or model it did not test, a comparison it did not run).
They must sound natural and on-topic, not absurd. Return a JSON object: {{"questions": ["...", "..."]}}"""


# ── Drafting ──────────────────────────────────────────────
def draft_paper(name: str):
    p, chunks = PAPERS[name], load_chunks(name)
    items, dropped, seen = [], [], set()

    def add(item: dict, quotes: list[str]):
        """Keep an item only if every evidence quote is found in the paper."""
        key = (item["type"], squash(" ".join(quotes)))
        if key in seen:
            dropped.append({**item, "reason": "duplicate", "quotes": quotes})
            return
        seen.add(key)
        groups = [find_evidence(chunks, q) for q in quotes]
        if not all(groups):
            dropped.append({**item, "reason": "evidence not found", "quotes": quotes})
            return
        item.update({"paper": name, "evidence": [[q] for q in quotes], "gold_chunks": groups})
        items.append(item)

    # 1) direct facts + near-identical pairs, one call per batch of passages
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        passages = "\n\n".join(f"[Passage {start + i}]\n{c}" for i, c in enumerate(batch))
        print(f"  {name}: passages {start}-{start + len(batch) - 1}")
        out = ask_json(FACT_PROMPT.format(title=p["title"], passages=passages))
        fact = out.get("fact")
        if fact and fact.get("question"):
            add({"type": "direct", "question": fact["question"], "answer": str(fact.get("answer", ""))},
                [fact.get("evidence", "")])
        conf = out.get("confusable")
        if conf and conf.get("question"):
            add({"type": "near_identical", "question": conf["question"], "answer": str(conf.get("answer", "")),
                 "wrong_answer": str(conf.get("wrong_answer", ""))}, [conf.get("evidence", "")])

    # number the base facts; their rewrites inherit the answer and evidence
    facts = [it for it in items if it["type"] == "direct"]
    for i, f in enumerate(facts):
        f["base_id"] = f"{name}-f{i}"

    # 2) four rewrites of every direct fact (types 2 to 5)
    for f in facts:
        print(f"  {name}: rewrites of {f['base_id']}")
        out = ask_json(VARIANT_PROMPT.format(title=p["title"], question=f["question"], answer=f["answer"]))
        for t in ("paraphrased", "verbose", "keyword", "plain_language"):
            if out.get(t):
                items.append({**{k: v for k, v in f.items() if k not in ("type", "question")},
                              "type": t, "question": out[t]})

    # 3) multi-hop questions built from pairs of facts
    by_id = {i: f for i, f in enumerate(facts)}
    listing = "\n".join(f"{i}. {f['question']} -> {f['answer']}" for i, f in by_id.items())
    print(f"  {name}: multi-hop")
    out = ask_json(MULTIHOP_PROMPT.format(title=p["title"], facts=listing, n=N_EXTRA))
    for q in out.get("questions", []):
        uses = [u for u in q.get("uses", []) if isinstance(u, int) and u in by_id]
        if len(set(uses)) == 2:
            a, b = (by_id[u] for u in uses)
            add({"type": "multi_hop", "question": q["question"], "answer": str(q.get("answer", "")),
                 "base_id": f"{a['base_id']}+{b['base_id']}"},
                [a["evidence"][0][0], b["evidence"][0][0]])

    # 4) unanswerable questions (checked by a human, since absence can't be verified by string match)
    print(f"  {name}: unanswerable")
    out = ask_json(UNANSWERABLE_PROMPT.format(
        title=p["title"], questions="\n".join(f"- {f['question']}" for f in facts), n=N_EXTRA))
    for q in out.get("questions", []):
        if isinstance(q, str) and q.strip():
            items.append({"type": "unanswerable", "question": q.strip(), "answer": "I don't know",
                          "paper": name, "evidence": [], "gold_chunks": []})

    for i, it in enumerate(items):
        it["id"] = f"{name}-{it['type']}-{i}"
    path = os.path.join(OUT_DIR, f"drafts_{name}.json")
    with open(path, "w") as f:
        json.dump({"items": items, "dropped": dropped}, f, indent=1)
    counts = {t: sum(it["type"] == t for it in items) for t in TYPES}
    print(f"\n{name}: kept {len(items)}, dropped {len(dropped)} (quote not found or duplicate)\n{counts}")
    print(f"Saved to {path}")


def write_csv():
    """One review sheet. Fill the keep column with y or n; edit question/answer text in place if needed."""
    path = os.path.join(OUT_DIR, "review.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "paper", "type", "question", "answer", "evidence", "wrong_answer", "keep", "note"])
        for name in PAPERS:
            draft = os.path.join(OUT_DIR, f"drafts_{name}.json")
            if not os.path.exists(draft):
                continue
            with open(draft) as d:
                for it in json.load(d)["items"]:
                    ev = " && ".join(g[0] for g in it["evidence"])
                    w.writerow([it["id"], it["paper"], it["type"], it["question"], it["answer"], ev,
                                it.get("wrong_answer", ""), "", ""])
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["chunks", "draft", "csv"])
    ap.add_argument("--paper", choices=list(PAPERS))
    args = ap.parse_args()
    if args.step == "chunks":
        build_chunks()
    elif args.step == "draft":
        if not args.paper:
            ap.error("draft needs --paper")
        if not _clients:
            ap.error("no GROQ_API_KEY / GROQ_JUDGE_API_KEY in .env")
        print(f"Drafting with {DRAFT_MODEL} using {len(_clients)} key(s)")
        draft_paper(args.paper)
    else:
        write_csv()


if __name__ == "__main__":
    main()