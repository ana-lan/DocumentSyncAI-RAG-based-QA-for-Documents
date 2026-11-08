# src/ingestion/cleaner.py
import re
from config import DROP_REFERENCES

# A line that is only "References" / "Bibliography", the usual heading of a paper's bibliography
_REF_HEADING = re.compile(r"\n\s*(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n")


def clean_text(text: str) -> str:
    """
    Clean raw extracted text from a single page.
    Returns: cleaned string
    """
    text = re.sub(pattern=r'-\n', repl="", string=text)
    text = re.sub(pattern=r'\n+', repl='\n', string=text)
    text = re.sub(pattern=r' +', repl=' ', string=text)
    text = text.strip()
    return text


def strip_references(text: str) -> str:
    """
    Cut everything from the References heading onward.

    Bibliography entries are full of paper titles that match many questions
    ("attention", "translation") but never contain an answer, so they steal
    retrieval slots. Only a heading in the second half of the document counts,
    so a stray "References" line early on never cuts the body.
    Note: an appendix after the references is dropped too.
    """
    for match in reversed(list(_REF_HEADING.finditer(text))):
        if match.start() > len(text) * 0.5:
            return text[: match.start()]
    return text


def clean_pages(pages: list[str], drop_references: bool = DROP_REFERENCES) -> str:
    """
    Clean all pages and join into one document string.
    Returns: single cleaned string for the whole document
    """
    cleaned_pages = []
    for page in pages:
        cleaned_text = clean_text(page)
        if len(cleaned_text) > 100:
            cleaned_pages.append(cleaned_text)
    text = "\n\n".join(cleaned_pages)
    if drop_references:
        text = strip_references(text)
    return text
