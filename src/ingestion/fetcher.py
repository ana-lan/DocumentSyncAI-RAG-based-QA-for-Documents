# src/ingestion/fetcher.py
import arxiv
import os
import json
import urllib.request
from config import ARXIV_QUERY, MAX_PAPERS, RAW_DIR

def fetch_papers():
    """
    Search ArXiv, download PDFs to RAW_DIR, save metadata as JSON.
    Returns: list of metadata dicts
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    search = arxiv.Search(
        query = ARXIV_QUERY,
        max_results = MAX_PAPERS,
    )

    client = arxiv.Client()
    counter = 0
    metadata = []

    for paper in client.results(search):
        counter += 1
        filename = f"{counter}.pdf"
        try:
            urllib.request.urlretrieve(paper.pdf_url, os.path.join(RAW_DIR, filename))
        except Exception as e:
            print(f"Skipping paper {counter} ({paper.title}) : {e}")
            continue
        metadata.append(
            {
                "id" : counter,
                "title" : paper.title,
                "authors" : [str(a) for a in paper.authors],
                "pdf_path" : os.path.join(RAW_DIR, filename)
            }
        )
        print(f"Saved paper id {counter} : {paper.title}")

    with open(os.path.join(RAW_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nDone. {counter} papers saved to {RAW_DIR}")

    return metadata

if __name__ == "__main__":
    fetch_papers()