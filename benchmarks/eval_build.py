"""
E1, Phase 3 part B: pick, review, finalize and freeze the eval set.

Steps (run from repo root):
  python -m benchmarks.eval_build pick          # balanced shortlist -> eval/week2/review_pick.csv
  python -m benchmarks.eval_build sessions      # draft follow-up sessions -> eval/week2/sessions_review.csv
      ... review both CSVs: keep = y or n, edit text in place ...
  python -m benchmarks.eval_build finalize      # -> eval_set.json, sessions.json
  python -m benchmarks.eval_build freeze        # retrieve once -> frozen_eval.json, frozen_sessions.json

Why freeze: E1 compares bf16 and fp8. If each model run did its own retrieval,
any difference in context would muddy the comparison. Retrieving once and saving
the top 5 chunks means both models answer from exactly the same passages.
"""
import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

from benchmarks.eval_draft import PAPERS, TYPES, OUT_DIR, load_chunks, find_evidence, ask_json
from benchmarks.metrics import hit_rank

PER_TYPE = 5                 # questions per paper per type in the final set (3 x 8 x 5 = 120)
SPARE = 1                    # extra picks per paper per type, so dropping one in review costs nothing
SESSION_LEN = 15             # follow-up questions per paper
SESSION_DRAFT = 18           # drafted per paper, so a few can be dropped
REWRITE_TYPES = ["direct", "paraphrased", "verbose", "keyword", "plain_language"]
SECTION_HINTS = {            # where each follow-up session starts: the paper's method section
    "attention": "Model Architecture",
    "rag": "Methods",
    "lora": "OUR METHOD",
}
COLUMNS = ["id", "paper", "type", "base_id", "question", "answer", "evidence", "wrong_answer", "keep", "note"]


def load_drafts(name: str) -> list[dict]:
    with open(os.path.join(OUT_DIR, f"drafts_{name}.json")) as f:
        return json.load(f)["items"]


def spread(items: list, k: int) -> list:
    """k items evenly spaced through the list, so picks cover the whole paper, not just the start."""
    if len(items) <= k:
        return items
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def evidence_text(item: dict) -> str:
    return " && ".join(g[0] for g in item.get("evidence", []))


def write_review(path: str, rows: list[dict], extra_cols: tuple = ()):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS + list(extra_cols))
        for it in rows:
            w.writerow([it["id"], it["paper"], it["type"], it.get("base_id", ""), it["question"],
                        it["answer"], evidence_text(it), it.get("wrong_answer", ""), "", ""]
                       + [it.get(c, "") for c in extra_cols])
    print(f"Saved {path} ({len(rows)} rows)")


# ── pick ──────────────────────────────────────────────────
def pick():
    """
    Shortlist PER_TYPE + SPARE per paper per type.
    Types 1 to 5 come as a block: choose base facts spread across the paper,
    then take every rewrite of those facts, so phrasing is the only thing that varies.
    """
    k = PER_TYPE + SPARE
    rows = []
    for name in PAPERS:
        items = load_drafts(name)
        by_type = defaultdict(list)
        for it in items:
            by_type[it["type"]].append(it)

        bases = sorted(by_type["direct"], key=lambda it: it["gold_chunks"][0][0])
        chosen = {b["base_id"] for b in spread(bases, k)}
        for t in REWRITE_TYPES:
            rows += [it for it in by_type[t] if it.get("base_id") in chosen]

        # true multi-hop first: the two quotes live in different chunks
        mh = sorted(by_type["multi_hop"],
                    key=lambda it: bool(set(it["gold_chunks"][0]) & set(it["gold_chunks"][1])))
        rows += mh[:k]
        near = sorted(by_type["near_identical"], key=lambda it: it["gold_chunks"][0][0])
        rows += spread(near, k)
        rows += by_type["unanswerable"][:k]

    # group a base fact's five versions together, which makes review faster
    order = {t: i for i, t in enumerate(TYPES)}
    rows.sort(key=lambda it: (it["paper"], it.get("base_id") if it["type"] in REWRITE_TYPES else "~" + it["type"],
                              order[it["type"]]))
    write_review(os.path.join(OUT_DIR, "review_pick.csv"), rows)
    counts = Counter((it["paper"], it["type"]) for it in rows)
    short = {key: n for key, n in counts.items() if n < PER_TYPE}
    print("Short of", PER_TYPE, "before review:", short or "none")


# ── sessions ──────────────────────────────────────────────
SESSION_PROMPT = """Below is one section of the paper "{title}".

{passages}

A reader is studying this section and asks {n} questions in a row, the way a curious person explores
a document: start broad, then dig into details, compare things, ask why.
Rules:
- Every question must make sense ON ITS OWN. The system answering has no memory of earlier
  questions, so never write "it", "that" or "the previous" to refer back.
- Every question must be answerable from ONE passage above.
Return a JSON object: {{"questions": [{{"question": "...", "answer": "...", "evidence": "..."}}]}}
where evidence is an exact quote of 5 to 15 words, copied character for character from the
passage that contains the answer."""


def section_window(name: str, chunks: list[str], width: int = 4) -> int:
    """First chunk of the method section (found by its heading), else the middle of the paper."""
    hint = SECTION_HINTS[name].lower()
    for i, c in enumerate(chunks):
        if i >= 2 and hint in c.lower():
            return min(i, len(chunks) - width)
    return max(0, len(chunks) // 2 - width // 2)


def draft_sessions():
    rows = []
    for name, p in PAPERS.items():
        chunks = load_chunks(name)
        start = section_window(name, chunks)
        window = chunks[start:start + 4]
        passages = "\n\n".join(f"[Passage {start + i}]\n{c}" for i, c in enumerate(window))
        print(f"  {name}: session on chunks {start}-{start + len(window) - 1}")
        out = ask_json(SESSION_PROMPT.format(title=p["title"], passages=passages, n=SESSION_DRAFT))
        turn = 0
        for q in out.get("questions", []):
            gold = find_evidence(chunks, q.get("evidence", ""))
            if not gold or not q.get("question"):
                print(f"    dropped (quote not found): {q.get('question', '')[:70]}")
                continue
            rows.append({"id": f"{name}-session-{turn}", "paper": name, "type": "session",
                         "session": name, "turn": turn, "question": q["question"],
                         "answer": str(q.get("answer", "")), "evidence": [[q["evidence"]]],
                         "gold_chunks": [gold]})
            turn += 1
        print(f"  {name}: {turn} kept")
    with open(os.path.join(OUT_DIR, "sessions_drafts.json"), "w") as f:
        json.dump({"items": rows}, f, indent=1)
    write_review(os.path.join(OUT_DIR, "sessions_review.csv"), rows)


# ── finalize ──────────────────────────────────────────────
def read_review(path: str) -> dict[str, dict]:
    with open(path, newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def apply_review(item: dict, row: dict) -> dict | None:
    """Return the item with the reviewer's edits, or None if dropped. Edited evidence is re-checked."""
    if row["keep"].strip().lower() not in ("y", "yes"):
        return None
    it = {**item, "question": row["question"].strip(), "answer": row["answer"].strip(),
          "note": row.get("note", "")}
    if row.get("wrong_answer"):
        it["wrong_answer"] = row["wrong_answer"].strip()
    quotes = [q.strip() for q in row["evidence"].split("&&") if q.strip()]
    if quotes != [g[0] for g in item.get("evidence", [])]:
        chunks = load_chunks(item["paper"])
        gold = [find_evidence(chunks, q) for q in quotes]
        if not all(gold):
            raise SystemExit(f"{item['id']}: edited evidence not found in the paper: {quotes}")
        it["evidence"], it["gold_chunks"] = [[q] for q in quotes], gold
    return it


def answer_kind(it: dict) -> str:
    """How E1 will score the answer: exact match for short numbers, a judge for text, refusal check."""
    if it["type"] == "unanswerable":
        return "refusal"
    a = it["answer"]
    return "number" if re.search(r"\d", a) and len(a) <= 25 else "text"


def finalize():
    review = read_review(os.path.join(OUT_DIR, "review_pick.csv"))
    drafts = {it["id"]: it for name in PAPERS for it in load_drafts(name)}
    kept = [it for i, row in review.items() if (it := apply_review(drafts[i], row))]

    final = []
    for name in PAPERS:
        mine = [it for it in kept if it["paper"] == name]
        # types 1 to 5: prefer base facts with all five versions kept, so the phrasing comparison is paired
        per_base = Counter(it["base_id"] for it in mine if it["type"] in REWRITE_TYPES)
        bases = [b for b, _ in per_base.most_common()][:PER_TYPE]
        for t in REWRITE_TYPES:
            final += [it for it in mine if it["type"] == t and it["base_id"] in bases][:PER_TYPE]
        for t in ("multi_hop", "near_identical", "unanswerable"):
            final += [it for it in mine if it["type"] == t][:PER_TYPE]
    for it in final:
        it["answer_kind"] = answer_kind(it)

    with open(os.path.join(OUT_DIR, "eval_set.json"), "w") as f:
        json.dump({"items": final}, f, indent=1)
    print(f"eval_set.json: {len(final)} questions")
    print(f"{'type':<16}" + "".join(f"{n:>11}" for n in PAPERS))
    counts = Counter((it["paper"], it["type"]) for it in final)
    for t in TYPES:
        line = "".join(f"{counts[(n, t)]:>11}" for n in PAPERS)
        flag = "" if all(counts[(n, t)] >= PER_TYPE for n in PAPERS) else "   <- short"
        print(f"{t:<16}{line}{flag}")

    spath = os.path.join(OUT_DIR, "sessions_review.csv")
    if os.path.exists(spath):
        sreview = read_review(spath)
        with open(os.path.join(OUT_DIR, "sessions_drafts.json")) as f:
            sdrafts = {it["id"]: it for it in json.load(f)["items"]}
        sess = [it for i, row in sreview.items() if (it := apply_review(sdrafts[i], row))]
        out = []
        for name in PAPERS:
            mine = sorted([it for it in sess if it["paper"] == name], key=lambda it: it["turn"])[:SESSION_LEN]
            out += [{**it, "answer_kind": answer_kind(it)} for it in mine]
        with open(os.path.join(OUT_DIR, "sessions.json"), "w") as f:
            json.dump({"items": out}, f, indent=1)
        print(f"\nsessions.json: {Counter(it['paper'] for it in out)}")


# ── freeze ────────────────────────────────────────────────
def freeze():
    """Retrieve the top 5 chunks for every question once, from its own paper, and save them."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    from config import EMBEDDING_MODEL, DEFAULT_TOP_K
    from src.retrieval.retriever import retrieve

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.EphemeralClient()
    for name in PAPERS:
        chunks = load_chunks(name)
        col = client.get_or_create_collection(name=f"frozen_{name}", metadata={"hnsw:space": "cosine"})
        col.add(ids=[f"chunk_{i}" for i in range(len(chunks))],
                embeddings=model.encode(chunks).tolist(), documents=chunks,
                metadatas=[{"chunk_index": i} for i in range(len(chunks))])

    def run(items: list[dict]) -> list[dict]:
        rows = []
        for it in items:
            got = retrieve(query=it["question"], strategy_name=f"frozen_{it['paper']}", client=client)
            texts = [g["text"] for g in got]
            ranks = [hit_rank(texts, group) for group in it.get("evidence", [])]
            # a multi-hop question only "hits" when every piece of evidence was retrieved
            hit = all(ranks) if ranks else None
            rows.append({**it,
                         "retrieved_chunk_ids": [g["chunk_index"] for g in got],
                         "retrieved_scores": [round(g["score"], 4) for g in got],
                         "retrieved_texts": texts,
                         "hit": hit, "hit_rank": max(ranks) if hit else None})
        return rows

    meta = {"embedding_model": EMBEDDING_MODEL, "top_k": DEFAULT_TOP_K, "chunking": "fixed_1024"}
    for src, dst in (("eval_set.json", "frozen_eval.json"), ("sessions.json", "frozen_sessions.json")):
        path = os.path.join(OUT_DIR, src)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = run(json.load(f)["items"])
        with open(os.path.join(OUT_DIR, dst), "w") as f:
            json.dump({"meta": meta, "rows": rows}, f, indent=1)
        print(f"\n{dst}: {len(rows)} questions")
        retrieval_table(rows)


def retrieval_table(rows: list[dict]):
    """hit@5 and MRR by question type. Unanswerable questions have no evidence, so they are skipped."""
    by_type = defaultdict(list)
    for r in rows:
        if r["hit"] is not None:
            by_type[r["type"]].append(r)
    print(f"{'type':<16}{'n':>4}{'hit@5':>8}{'MRR':>7}")
    for t, rs in by_type.items():
        hit = sum(r["hit"] for r in rs) / len(rs)
        mrr = sum(1 / r["hit_rank"] for r in rs if r["hit"]) / len(rs)
        print(f"{t:<16}{len(rs):>4}{hit:>8.2f}{mrr:>7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pick", "sessions", "finalize", "freeze"])
    step = ap.parse_args().step
    {"pick": pick, "sessions": draft_sessions, "finalize": finalize, "freeze": freeze}[step]()


if __name__ == "__main__":
    main()