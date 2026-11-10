# src/ingestion/parser.py
import pdfplumber
import os
from config import RAW_DIR, PROCESSED_DIR, PDF_X_TOLERANCE


def parse_pdf(pdf, x_tolerance: float = PDF_X_TOLERANCE) -> list[str]:
    """
    Extract text from a PDF, one string per page.

    pdf can be a file path or an open file (the Streamlit upload), so the app,
    the pipeline and the benchmarks all extract text the same way.

    x_tolerance: if the gap between two characters is smaller than this
    (in PDF points), pdfplumber treats them as one word. Too high glues
    words together ("theTransformer"), too low splits them ("Trans former").

    Returns: list of page strings
    """
    content = []
    with pdfplumber.open(pdf) as doc:
        for page in doc.pages:
            text = page.extract_text(x_tolerance=x_tolerance)
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
