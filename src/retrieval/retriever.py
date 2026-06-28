# src/retrieval/retriever.py
from src.embedding.embedder import get_embedding_model, get_chroma_client
from config import DEFAULT_TOP_K, SIMILARITY_THRESHOLD

def retrieve(query: str, strategy_name: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Embed query, search ChromaDB collection for strategy_name,
    filter by similarity threshold, return top results.
    
    Returns: list of dicts with keys: {text, score, paper_id, chunk_index}
    """
    model = get_embedding_model()
    client = get_chroma_client()

    embedding = model.encode(query).tolist()
    collection = client.get_collection(name=strategy_name)

    results = collection.query(
        query_embeddings=[embedding], 
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []
    for text, meta, dist in zip(documents, metadatas, distances):
        score = 1 - dist
        if score >= SIMILARITY_THRESHOLD:
            retrieved.append({
                "text" : text,
                "score" : score,
                "paper_id" : meta["paper_id"],
                "chunk_index" : meta["chunk_index"]
            })
    return retrieved