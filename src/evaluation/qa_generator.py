# src/evaluation/qa_generator.py
import json
import os
from langchain_groq import ChatGroq
from src.embedding.embedder import get_chroma_client
from config import QA_GEN_MODEL, QA_PAIRS_DIR, NUM_QA_PAIRS, CHUNK_STRATEGIES
from dotenv import load_dotenv

load_dotenv()

def generate_qa_from_chunk(chunk_text: str, paper_id: int, llm) -> dict | None:
    """
    Given a chunk of text, use LLM to generate one Q&A pair.
    Returns: dict {question, answer, context, paper_id} or None if failed
    """
    prompt = f"""Given this text: {chunk_text}
    Generate one question and answer. Respond ONLY in JSON with no extra text:
    {{"question": "...", "answer": "..."}}"""
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content.strip())
        return {
            "question" : parsed["question"],
            "answer" : parsed["answer"],
            "context" : chunk_text,
            "paper_id" : paper_id
        }
    except Exception as e:
        print(f"Failed: {e}")
        return None

def generate_qa_pairs(strategy_name: str = "fixed_512") -> list[dict]:
    """
    Pull chunks from ChromaDB, generate Q&A pairs, save to JSON.
    Returns: list of Q&A pair dicts
    """
    os.makedirs(QA_PAIRS_DIR, exist_ok=True)

    llm = ChatGroq(
        model = QA_GEN_MODEL,
        temperature = 0.1
    )

    client = get_chroma_client()
    collection = client.get_collection(name=strategy_name)
    results = collection.get(limit=NUM_QA_PAIRS)

    qa_pairs = []
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        qa = generate_qa_from_chunk(chunk_text=doc, paper_id=meta["paper_id"], llm=llm)
        if qa is None:
            continue
        qa_pairs.append(qa)
        print(f"Generated {i+1}/{NUM_QA_PAIRS}: paper {meta['paper_id']}")
    
    with open(os.path.join(QA_PAIRS_DIR, f"{strategy_name}_qa_pairs.json"), "w") as f:
        json.dump(qa_pairs, f, indent=2)
    
    print(f"\nDone. {len(qa_pairs)} Q&A pairs saved.")
    return qa_pairs

if __name__ == "__main__":
    for strategy in CHUNK_STRATEGIES.keys():
        print(f"\nGenerating Q&A pairs for {strategy}...")
        generate_qa_pairs(strategy_name=strategy)