# src/generation/generator.py
import os
import time
from config import TOGETHER_MODEL, MAX_TOKENS, TEMPERATURE
from dotenv import load_dotenv
from langchain_together import ChatTogether
from openai import RateLimitError

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

    llm = ChatTogether(
        model=TOGETHER_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )

    start = time.time()
    response = None

    for attempt in range(5):
        try:
            response = llm.invoke(prompt)
            break
        except RateLimitError as e:
            if attempt < 4:
                wait = 90 * (attempt+1)
                print(f"Rate limit hit, waiting {wait}s before retry {attempt+1}/5...")
                time.sleep(wait)
            else:
                raise e
        except Exception as e:
            if "rate" in str(e).lower() and attempt < 4:
                wait = 90 * (attempt+1)
                print(f"Rate limit hit, waiting {wait}s before retry {attempt+1}/5...")
                time.sleep(wait)
            else:
                raise e
    
    if response is None:
        raise RuntimeError("Failed to get response after 5 attempts")
    latency = time.time() - start

    return {
    "answer": response.content,
    "latency_seconds": latency,
    "num_chunks_used": len(retrieved_chunks)
    }