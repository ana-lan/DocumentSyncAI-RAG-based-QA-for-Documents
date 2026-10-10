# src/generation/generator.py
import time
from functools import lru_cache
from langchain_openai import ChatOpenAI
from openai import RateLimitError
from config import LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, MAX_TOKENS, TEMPERATURE


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks + query into a grounded prompt."""
    context_block = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    prompt = f"""Context:
    {context_block}
    Question: {query}
    Instructions: Answer only from the context above. If the answer is not in the context, say "I don't know".
    Answer:"""
    return prompt


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Build the LLM client once and reuse it (keeps HTTP connections warm)."""
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        timeout=120,
        max_retries=0,   # our own retry loop below handles retries
    )


def generate(query: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generate an answer from retrieved chunks using whichever
    OpenAI-compatible backend LLM_BASE_URL points to (Together or vLLM).
    Returns: dict with keys {answer, latency_seconds, num_chunks_used, backend}
    """
    prompt = build_prompt(query=query, retrieved_chunks=retrieved_chunks)
    llm = get_llm()

    start = time.time()
    response = None

    for attempt in range(5):
        try:
            response = llm.invoke(prompt)
            break
        except RateLimitError as e:
            if attempt < 4:
                wait = 90 * (attempt + 1)
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
        "num_chunks_used": len(retrieved_chunks),
        "backend": LLM_BASE_URL,
    }