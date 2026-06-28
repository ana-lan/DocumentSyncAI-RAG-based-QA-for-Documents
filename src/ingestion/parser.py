# src/ingestion/parser.py
import pdfplumber
import os
from config import RAW_DIR, PROCESSED_DIR

def parse_pdf(pdf_path: str) -> list[str]:
    """
    Extract text from a PDF file, one string per page.
    Returns: list of page strings
    """
    content = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                content.append(text)
    return content

def parse_all(metadata: list[dict]) -> dict:
    """
    Parse all PDFs listed in metadata.
    Returns: dict of {paper_id: [page1_text, page2_text, ...]}
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    papers = {}

    for doc in metadata:
        doc_content = parse_pdf(doc["pdf_path"])
        papers[doc["id"]] = doc_content
        with open(os.path.join(PROCESSED_DIR, f"{doc['id']}.txt"), "w") as f:
            f.write("\n".join(doc_content))
        print(f"Parsed paper {doc['id']}: {doc['title']}")
    return papers

if __name__ == "__main__":
    import json
    with open("data/raw/metadata.json") as f:
        metadata = json.load(f)
    parse_all(metadata)