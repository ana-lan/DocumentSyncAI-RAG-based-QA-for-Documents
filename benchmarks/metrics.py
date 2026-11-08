"""
Scoring helpers shared by the week 2 benchmarks (e2e.py now, quality_eval.py later).

Matching is done on "squashed" text: lowercase, unicode-normalized, all
whitespace removed. That way "28.4 BLEU", "28.4BLEU" and "28. 4 BLEU" all
match, so a retrieval metric isn't fooled by how the PDF happened to space
its words. Retrieval quality and extraction quality are measured separately.
"""
import unicodedata


def squash(text: str) -> str:
    """Lowercase, NFKC-normalize (fixes ligatures like 'ﬁ'), drop all whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(text.split())


def contains_any(text: str, phrases: list[str]) -> bool:
    """True if any phrase appears in text (whitespace-insensitive)."""
    t = squash(text)
    return any(squash(p) in t for p in phrases)


def hit_rank(chunk_texts: list[str], evidence: list[str]) -> int | None:
    """1-based rank of the first retrieved chunk containing the gold evidence, or None."""
    for rank, text in enumerate(chunk_texts, start=1):
        if contains_any(text, evidence):
            return rank
    return None


def reciprocal_rank(rank: int | None) -> float:
    """MRR term: 1.0 if the evidence was the top chunk, 0.5 if second, ..., 0 if missing."""
    return 1.0 / rank if rank else 0.0


def is_refusal(answer: str) -> bool:
    a = answer.lower().replace("’", "'")
    return "don't know" in a or "do not know" in a or "cannot be determined" in a


def answer_matches(answer: str, key: list[list[str]]) -> bool:
    """
    First-pass correctness check. key is a list of groups; every group must
    match, and a group matches if any of its alternatives appears.
    Example: [["4", "four"], ["0.6"]] needs "4" or "four" AND "0.6".
    A refusal never counts as correct. Misses still get a human look.
    """
    if is_refusal(answer):
        return False
    return all(contains_any(answer, group) for group in key)


def citation_correct(citations: list[int], chunk_texts: list[str], evidence: list[str]) -> bool | None:
    """
    True if at least one cited passage actually contains the gold evidence.
    None when the answer cited nothing (scored separately as citation rate).
    Out-of-range numbers (e.g. [7] when 5 chunks were given) count as wrong.
    """
    if not citations:
        return None
    for n in citations:
        if 1 <= n <= len(chunk_texts) and contains_any(chunk_texts[n - 1], evidence):
            return True
    return False
