"""
Phase 1.1: compare pdfplumber x_tolerance settings on one PDF.

x_tolerance is the largest gap (in PDF points) between two characters that
pdfplumber still treats as the same word. Too high and words glue together
("thedotproducts"); too low and words break apart ("Trans former").

For each setting this prints:
  glued   tokens of 18+ letters, which are almost always two or more words stuck together
  intact  how often a few common words appear whole; gluing AND splitting both lower it
  sample  the same sentence at each setting, so you can see the difference by eye

Pick the setting with the most intact words and the fewest glued ones.

Usage (from repo root):
  python -m benchmarks.diagnose_extraction
  python -m benchmarks.diagnose_extraction --pdf data/raw/3.pdf --find "learning rate"
"""
import argparse
import re

from src.ingestion.parser import parse_pdf
from src.ingestion.cleaner import clean_pages

TOLERANCES = [3, 2, 1.5, 1]
COMMON_WORDS = ["attention", "the", "model", "layer", "layers", "sequence",
                "translation", "training", "encoder", "decoder", "positions", "with"]


def analyse(text: str) -> dict:
    tokens = re.findall(r"[A-Za-z]+", text)
    lower = [t.lower() for t in tokens]
    glued = [t for t in tokens if len(t) >= 18]
    return {
        "tokens": len(tokens),
        "glued": len(glued),
        "intact": sum(lower.count(w) for w in COMMON_WORDS),
        "glued_examples": sorted(set(glued), key=len, reverse=True)[:6],
    }


def snippet(text: str, needle: str, width: int = 220) -> str:
    """The text around the first occurrence of needle, ignoring spacing differences."""
    squashed_needle = needle.replace(" ", "").lower()
    # map each non-space character back to its index in the original text
    positions = [i for i, ch in enumerate(text) if not ch.isspace()]
    squashed = "".join(text[i] for i in positions).lower()
    hit = squashed.find(squashed_needle)
    if hit < 0:
        return "(not found)"
    start = positions[hit]
    return text[max(0, start - 40): start + width].replace("\n", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="benchmarks/attention.pdf")
    parser.add_argument("--find", default="parallel attention layers",
                        help="phrase to show at every setting for an eyeball comparison")
    args = parser.parse_args()

    results = {}
    for tol in TOLERANCES:
        text = clean_pages(parse_pdf(args.pdf, x_tolerance=tol), drop_references=False)
        results[tol] = (analyse(text), text)

    print(f"\n{'x_tolerance':>11} {'tokens':>7} {'glued':>6} {'intact':>7}")
    for tol, (a, _) in results.items():
        print(f"{tol:>11} {a['tokens']:>7} {a['glued']:>6} {a['intact']:>7}")

    for tol, (a, text) in results.items():
        print(f"\n--- x_tolerance={tol} ---")
        print(f"glued examples: {a['glued_examples']}")
        print(f"sample: {snippet(text, args.find)}")

    best = max(results, key=lambda t: (results[t][0]["intact"], -results[t][0]["glued"]))
    print(f"\nMost intact words: x_tolerance={best}. Check the samples agree before using it.")


if __name__ == "__main__":
    main()
