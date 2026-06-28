# src/ingestion/cleaner.py
import re

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

def clean_pages(pages: list[str]) -> str:
    """
    Clean all pages and join into one document string.
    Returns: single cleaned string for the whole document
    """
    cleaned_pages = []
    for page in pages:
        cleaned_text = clean_text(page)
        if len(cleaned_text) > 100:
            cleaned_pages.append(cleaned_text)
    return "\n\n".join(cleaned_pages)