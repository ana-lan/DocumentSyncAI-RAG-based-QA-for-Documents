"""
E1 / E2 scoring: grade the answers collected by quality_eval.py.

For each answer:
  correct    numbers: every headline number of the gold answer appears in the answer
             text: an LLM judge compares the meaning with the gold answer
             unanswerable: correct only if the model refused ("I don't know")
  supported  faithfulness, LLM judge: every claim is backed by the passages the model was shown
  citation   the passage(s) the model cited really contain the gold evidence

Judge verdicts are cached in eval/week2/judge_cache.json, keyed by the exact question,
answer and passages. bf16 and fp8 often give identical answers, so the second model
costs few new judge calls, and a run interrupted by rate limits resumes where it stopped.

Usage (from repo root):
  python -m benchmarks.score_eval benchmarks/results/week2/quality_e1_bf16.json
  python -m benchmarks.score_eval <bf16 file> <fp8 file> --compare     # adds McNemar's test
"""
import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict

from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

from config import LLM_BASE_URL, LLM_MODEL
from benchmarks.metrics import is_refusal, contains_any

CACHE_PATH = "eval/week2/judge_cache.json"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)

# ── numbers ───────────────────────────────────────────────
_NUM = re.compile(r"\d+(?:\.\d+)?")


def numbers(text: str) -> set[float]:
    """All numbers in a text: '100,000 steps' -> {100000}; '2.3 · 10^19' -> {2.3, 10, 19}."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)     # 100,000 -> 100000
    # powers of ten: "10^19", and "1019" where PDF extraction dropped the superscript -> "10 19"
    text = re.sub(r"([·×]\s*)10\^?(\d{1,2})\b", r"\g<1>10 \2", text)
    return {float(n) for n in _NUM.findall(text)}


def headline_numbers(gold: str) -> set[float]:
    """Numbers the answer must contain: ignore anything in brackets or after ± (explanations, error bars)."""
    core = re.sub(r"\(.*?\)", "", gold)
    core = re.sub(r"±\s*\S+", "", core)
    return numbers(core)


def number_correct(row: dict) -> bool:
    if is_refusal(row["model_answer"]):
        return False
    got = numbers(row["model_answer"])
    need = headline_numbers(row["gold_answer"])
    if not need or not need <= got:
        return False
    # near-identical: hedging with the look-alike value doesn't count
    wrong = numbers(row.get("wrong_answer") or "") - need
    return not (row["type"] == "near_identical" and wrong and wrong <= got)


# ── judge ─────────────────────────────────────────────────
JUDGE_PROMPT = """You are grading an answer from a question-answering system over a research paper.

Question: {question}
Reference answer: {gold}
System answer: {answer}

Passages the system was given:
{passages}

Return a JSON object with two keys:
"correct": true if the system answer gives the same answer as the reference. Wording may differ and
           extra correct detail is fine. A different value, a missing part, or a refusal is false.
"supported": true if every factual claim in the system answer is stated in, or follows directly from,
             the passages above. false if it states anything the passages do not contain.
Return only the JSON object."""

_keys = [k for k in (os.getenv("GROQ_API_KEY"), os.getenv("GROQ_JUDGE_API_KEY")) if k]
_clients = [OpenAI(base_url=LLM_BASE_URL, api_key=k, max_retries=0, timeout=90) for k in _keys]
_turn = 0


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=0)


def judge(row: dict, cache: dict) -> dict:
    """One judge call per unique (question, gold, answer, passages); verdicts are cached."""
    global _turn
    # The key uses the SET of passages (sorted): whether an answer is correct and supported doesn't
    # depend on passage order. Checked with benchmarks/check_judge_order.py: the 3 E2 answers that
    # reused a verdict across passage orders got the same verdict when re-judged in their real order.
    key = hashlib.sha256(json.dumps([row["question"], row["gold_answer"], row["model_answer"],
                                     sorted(row["shown_texts"]), JUDGE_MODEL]).encode()).hexdigest()
    if key in cache:
        return cache[key]
    passages = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(row["shown_texts"], 1))
    prompt = JUDGE_PROMPT.format(question=row["question"], gold=row["gold_answer"],
                                 answer=row["model_answer"], passages=passages)
    for attempt in range(12):
        client = _clients[_turn % len(_clients)]
        _turn += 1
        try:
            reply = client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0, max_tokens=300,
                messages=[{"role": "user", "content": prompt}]).choices[0].message.content
        except RateLimitError as e:
            wait = 15 * (attempt + 1)
            print(f"    {row['id']}: rate limit, waiting {wait}s ({str(e)[:150]})", flush=True)
            time.sleep(wait)
            continue
        except (APITimeoutError, APIConnectionError) as e:
            print(f"    {row['id']}: no answer from Groq ({type(e).__name__}), retrying", flush=True)
            continue
        reply = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S)
        try:
            verdict = json.loads(reply[reply.find("{"): reply.rfind("}") + 1])
            out = {"correct": bool(verdict["correct"]), "supported": bool(verdict["supported"])}
            cache[key] = out
            save_cache(cache)                 # save every verdict, so a crash or Ctrl+C loses nothing
            return out
        except (ValueError, KeyError):
            print(f"    {row['id']}: judge reply unreadable, retrying: {reply[:120]!r}", flush=True)
            continue
    raise RuntimeError(f"judge failed for {row['id']}")


# ── citations ─────────────────────────────────────────────
def citation_ok(row: dict):
    """Every evidence piece must sit in some cited passage. None if nothing was cited or there's no evidence."""
    cited = [row["shown_texts"][n - 1] for n in row["citations"] if 1 <= n <= len(row["shown_texts"])]
    if not row["citations"] or not row.get("evidence"):
        return None
    return all(any(contains_any(t, group) for t in cited) for group in row["evidence"])


# ── stats ─────────────────────────────────────────────────
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion; behaves well for small n and rates near 0 or 1."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(a: dict, b: dict) -> dict:
    """
    Exact McNemar test on paired right/wrong outcomes. Only questions where the two
    models disagree carry information: b10 = A right, B wrong; b01 = A wrong, B right.
    Under "no difference", each disagreement is a coin flip.
    """
    ids = sorted(set(a) & set(b))
    b10 = sum(a[i] and not b[i] for i in ids)
    b01 = sum(b[i] and not a[i] for i in ids)
    n = b10 + b01
    tail = sum(math.comb(n, k) for k in range(0, min(b10, b01) + 1)) / 2 ** n if n else 1.0
    return {"paired": len(ids), "a_only_correct": b10, "b_only_correct": b01,
            "p_value": round(min(1.0, 2 * tail), 4)}


# ── scoring ───────────────────────────────────────────────
def score_file(path: str, cache: dict, faithfulness: bool) -> dict:
    with open(path) as f:
        data = json.load(f)
    rows = data["rows"]
    for i, r in enumerate(rows, 1):
        r["refused"] = is_refusal(r["model_answer"])
        r["citation_ok"] = citation_ok(r)
        kind = r.get("answer_kind", "text")
        if kind == "refusal":
            r["correct"] = r["refused"]
        elif kind == "number":
            r["correct"] = number_correct(r)
        needs_judge = not r["refused"] and (kind == "text" or faithfulness)
        if needs_judge:
            v = judge(r, cache)
            if kind == "text":
                r["correct"] = v["correct"]
            r["supported"] = v["supported"] if faithfulness else None
        else:
            if kind == "text":
                r["correct"] = False          # text answer that refused
            r["supported"] = None
        if i % 5 == 0:
            save_cache(cache)
            print(f"  scored {i}/{len(rows)}", flush=True)
    save_cache(cache)

    out_path = path.replace("quality_", "scored_")
    with open(out_path, "w") as f:
        json.dump({"meta": data["meta"], "rows": rows}, f, indent=1)
    return {"label": data["meta"]["label"], "rows": rows, "path": out_path}


def report(s: dict):
    rows = s["rows"]
    n = len(rows)
    k = sum(r["correct"] for r in rows)
    lo, hi = wilson(k, n)
    print(f"\n=== {s['label']} ===  accuracy {k}/{n} = {k/n:.1%}  (95% CI {lo:.1%} to {hi:.1%})")

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    print(f"{'type':<16}{'n':>4}{'acc':>7}{'95% CI':>17}")
    for t, rs in by_type.items():
        kk = sum(r["correct"] for r in rs)
        lo, hi = wilson(kk, len(rs))
        print(f"{t:<16}{len(rs):>4}{kk/len(rs):>7.0%}   {lo:>5.0%} to {hi:>4.0%}")

    answerable = [r for r in rows if r["type"] != "unanswerable" and r.get("hit") is not None]
    hit = [r for r in answerable if r["hit"]]
    miss = [r for r in answerable if not r["hit"]]
    if hit:
        print(f"\nevidence retrieved ({len(hit)}): accuracy {sum(r['correct'] for r in hit)/len(hit):.0%}")
    if miss:
        refused = sum(r["refused"] for r in miss)
        right = sum(r["correct"] for r in miss)
        wrong = len(miss) - refused - right
        print(f"evidence NOT retrieved ({len(miss)}): refused {refused/len(miss):.0%} (honest), "
              f"answered right anyway {right/len(miss):.0%}, answered WRONG {wrong/len(miss):.0%} (hallucination risk)")
    unans = [r for r in rows if r["type"] == "unanswerable"]
    if unans:
        print(f"unanswerable ({len(unans)}): correctly refused {sum(r['refused'] for r in unans)/len(unans):.0%}")

    sup = [r for r in rows if r.get("supported") is not None]
    if sup:
        print(f"faithfulness (answers backed by their passages): {sum(r['supported'] for r in sup)/len(sup):.0%} of {len(sup)}")
    answered = [r for r in rows if not r["refused"] and r.get("evidence")]
    cited = [r for r in answered if r["citation_ok"] is not None]
    if answered:
        print(f"citations: {len(cited)/len(answered):.0%} of answers cite a passage; "
              f"{sum(r['citation_ok'] for r in cited)/max(len(cited),1):.0%} of those cite the right one")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="quality_*.json files from quality_eval.py")
    ap.add_argument("--compare", action="store_true", help="McNemar test between the first two files")
    ap.add_argument("--no-faithfulness", action="store_true",
                    help="skip the faithfulness judge (fewer LLM calls; numbers need no judge then)")
    args = ap.parse_args()
    if not _clients:
        ap.error("no GROQ_API_KEY / GROQ_JUDGE_API_KEY in .env")

    cache = load_cache()
    print(f"Judge: {JUDGE_MODEL} | cached verdicts: {len(cache)}")
    scored = []
    for path in args.files:
        print(f"\nScoring {path}")
        scored.append(score_file(path, cache, not args.no_faithfulness))
    for s in scored:
        report(s)

    if args.compare and len(scored) >= 2:
        a, b = scored[0], scored[1]
        ca = {r["id"]: r["correct"] for r in a["rows"]}
        cb = {r["id"]: r["correct"] for r in b["rows"]}
        m = mcnemar(ca, cb)
        same = sum(ra["model_answer"] == rb["model_answer"] for ra, rb in
                   zip(sorted(a["rows"], key=lambda r: r["id"]), sorted(b["rows"], key=lambda r: r["id"])))
        print(f"\n=== {a['label']} vs {b['label']} ===")
        print(f"identical answers, word for word: {same}/{m['paired']}")
        print(f"only {a['label']} right: {m['a_only_correct']} | only {b['label']} right: {m['b_only_correct']}")
        print(f"McNemar exact p = {m['p_value']}  (above 0.05: no evidence the two differ in accuracy)")
        with open(a["path"].replace("scored_", "compare_"), "w") as f:
            json.dump({"a": a["label"], "b": b["label"], **m, "identical_answers": same}, f, indent=1)


if __name__ == "__main__":
    main()