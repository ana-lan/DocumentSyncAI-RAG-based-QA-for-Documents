# src/generation/generator.py
import os
import time
from langchain_groq import ChatGroq
from config import GROQ_MODEL, MAX_TOKENS, TEMPERATURE
from dotenv import load_dotenv

load_dotenv()

def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Format retrieved chunks + query into a grounded prompt.
    Returns: prompt string
    """
    context_block = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    prompt = f"""Context:
    {context_block}
    Question: {query}
    Instructions: Answer only from the context above. If the answer is not in the context, say "I don't know".
    Answer:"""
    return prompt

def generate(query: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generate an answer from retrieved chunks using Groq LLM.
    Returns: dict with keys {answer, latency_seconds, num_chunks_used}
    """
    prompt = build_prompt(query=query, retrieved_chunks=retrieved_chunks)

    llm = ChatGroq(
        model_name=GROQ_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )

    start = time.time()
    response = llm.invoke(prompt)
    latency = time.time() - start

    return {
    "answer": response.content,
    "latency_seconds": latency,
    "num_chunks_used": len(retrieved_chunks)
    }