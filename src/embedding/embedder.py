# src/embedding/embedder.py
from sentence_transformers import SentenceTransformer
import chromadb
from config import EMBEDDING_MODEL, VECTORSTORE_DIR

def get_embedding_model():
    """Load and return the embedding model."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model

def get_chroma_client():
    """Initialize and return a persistent ChromaDB client."""
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    return client

def embed_and_store(chunks: list[str], strategy_name: str, paper_id: int, model, client):
    """
    Embed chunks and store in ChromaDB under the strategy's collection.
    Each chunk stored with metadata: {paper_id, chunk_index}
    """
    collection = client.get_or_create_collection(name=strategy_name)
    embeddings = model.encode(chunks)

    for i, chunk in enumerate(chunks):
        collection.add(
            ids = [f"{paper_id}_{strategy_name}_{i}"],
            embeddings = [embeddings[i].tolist()],
            documents = [chunk],
            metadatas = [{"paper_id": paper_id, "chunk_index": i}]
        )
    return collection

def embed_all(parsed_papers: dict, chunked_papers: dict):
    """
    For every paper and every strategy, embed and store chunks.
    parsed_papers: {paper_id: [pages]}
    chunked_papers: {paper_id: {strategy_name: [chunks]}}
    """
    model = get_embedding_model()
    client = get_chroma_client()
    for paper_id in parsed_papers:
        for strategy in chunked_papers[paper_id]:
            chunks = chunked_papers[paper_id][strategy]
            embed_and_store(chunks, strategy, paper_id, model, client)
            print(f"Embedded paper {paper_id} with strategy {strategy} — {len(chunks)} chunks")