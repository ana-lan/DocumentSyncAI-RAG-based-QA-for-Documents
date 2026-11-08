# src/generation/generator.py
import re
import time
from functools import lru_cache
from langchain_openai import ChatOpenAI
from openai import RateLimitError
from config import (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, MAX_TOKENS, TEMPERATURE, PROMPT_STYLE,
                    PROMPT_LAYOUT, CHUNK_ORDER)


CITE_INSTRUCTIONS = (
    "Instructions: Answer only from the numbered context passages. "
    "After each fact, cite the passage it came from in square brackets, like [2]. "
    'If the answer is not in the context, say "I don\'t know" and cite nothing.'
)


def order_chunks(retrieved_chunks: list[dict], order: str = CHUNK_ORDER) -> list[dict]:
    """
    "score":    most relevant chunk first (how retrieval returns them).
    "position": chunks in the order they appear in the document. Two questions that
                retrieve the same chunks then build the same prompt start, which is
                what vLLM's prefix cache can reuse.
    """
    if order == "position":
        return sorted(retrieved_chunks, key=lambda c: c.get("chunk_index", 0))
    return list(retrieved_chunks)


def build_prompt(query: str, retrieved_chunks: list[dict], style: str = PROMPT_STYLE,
                 layout: str = PROMPT_LAYOUT, order: str = CHUNK_ORDER) -> str:
    """
    Format retrieved chunks + query into a grounded prompt.

    style="plain": the week 1 prompt, kept unchanged so before/after runs compare fairly.
    style="cite":  chunks are numbered [1]..[k] and the model must cite the ones it used.

    layout (cite only), for E2:
      "context_first":      Context, Question, Instructions  (variant A, the app today)
      "instructions_first": Instructions, Context, Question  (variants B and C)
    The fixed instructions come first in B and C, so every prompt shares that start.
    The question always comes after the chunks, so it never breaks a shared prefix.
    """
    if style == "plain":
        context_block = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
        prompt = f"""Context:
    {context_block}
    Question: {query}
    Instructions: Answer only from the context above. If the answer is not in the context, say "I don't know".
    Answer:"""
        return prompt

    chunks = order_chunks(retrieved_chunks, order)
    context_block = "\n\n".join(f"[{i}] {chunk['text']}" for i, chunk in enumerate(chunks, start=1))
    if layout == "instructions_first":
        return f"{CITE_INSTRUCTIONS}\n\nContext:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Instructions: Answer only from the numbered context passages above. "
        "After each fact, cite the passage it came from in square brackets, like [2]. "
        'If the answer is not in the context, say "I don\'t know" and cite nothing.\n\n'
        "Answer:"
    )


_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(answer: str) -> list[int]:
    """Pull cited passage numbers out of an answer: "... [2][4]" or "[2, 4]" -> [2, 4]."""
    cited = []
    for group in _CITATION.findall(answer):
        for n in group.split(","):
            n = int(n.strip())
            if n not in cited:
                cited.append(n)
    return cited


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


def generate(query: str, retrieved_chunks: list[dict], prompt_style: str = PROMPT_STYLE) -> dict:
    """
    Generate an answer from retrieved chunks using whichever
    OpenAI-compatible backend LLM_BASE_URL points to (Groq or vLLM).
    Returns: dict with keys {answer, citations, latency_seconds, num_chunks_used, backend}
    """
    prompt = build_prompt(query=query, retrieved_chunks=retrieved_chunks, style=prompt_style)
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
        "citations": extract_citations(response.content) if prompt_style == "cite" else [],
        "latency_seconds": latency,
        "num_chunks_used": len(retrieved_chunks),
        "backend": LLM_BASE_URL,
    }