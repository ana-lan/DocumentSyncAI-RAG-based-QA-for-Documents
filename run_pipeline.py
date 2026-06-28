# run_pipeline.py
import os
from src.ingestion.fetcher import fetch_papers
from src.ingestion.parser import parse_all
from src.ingestion.cleaner import clean_pages
from src.ingestion.chunker import chunk_document
from src.embedding.embedder import embed_all
from src.retrieval.retriever import retrieve
from src.generation.generator import generate
from config import DEFAULT_TOP_K, VECTORSTORE_DIR

def run_pipeline():
    if not os.path.exists(VECTORSTORE_DIR) or not os.listdir(VECTORSTORE_DIR):
        metadata = fetch_papers()
        papers = parse_all(metadata)

        chunked_papers = {}
        for paper_id, pages in papers.items():
            cleaned_text = clean_pages(pages)
            chunked_papers[paper_id] = chunk_document(cleaned_text)

        embed_all(parsed_papers=papers, chunked_papers=chunked_papers)
    else:
        print("Vectorstore already exists, skipping ingestion and embedding")

    query = "What are the challenges of machine learning in healthcare?"
    
    retrieved_chunks = retrieve(query=query, strategy_name="fixed_512", top_k=DEFAULT_TOP_K)
    result = generate(query=query, retrieved_chunks=retrieved_chunks)
    
    print(f"Answer: {result['answer']}")
    print(f"Latency: {result['latency_seconds']:.2f}s")
    print(f"Chunks used: {result['num_chunks_used']}")
    for chunk in retrieved_chunks:
        print(f"  Score: {chunk['score']:.3f} | Paper: {chunk['paper_id']}")

if __name__ == "__main__":
    run_pipeline()