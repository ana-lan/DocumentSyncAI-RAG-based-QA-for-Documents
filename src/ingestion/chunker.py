# src/ingestion/chunker.py
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_STRATEGIES

nltk.download("punkt", quiet=True)

def chunk_fixed(text: str, strategy_name: str) -> list[str]:
    """
    Split text using fixed token size + overlap.
    Returns: list of chunk strings
    """
    chunk_size, chunk_overlap = CHUNK_STRATEGIES[strategy_name]["chunk_size"], CHUNK_STRATEGIES[strategy_name]["chunk_overlap"]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def chunk_sentence(text: str, window: int = 5) -> list[str]:
    """
    Split text on sentence boundaries, group into windows of `window` sentences.
    Returns: list of chunk strings
    """
    chunks = []
    sentences = nltk.sent_tokenize(text)
    for i in range(0, len(sentences), window-1):
        chunk = sentences[i:i+window]
        chunk_str = " ".join(chunk)
        chunks.append(chunk_str)
    return chunks

def chunk_document(text: str) -> dict[str, list[str]]:
    """
    Apply all 4 strategies to a document.
    Returns: dict of {strategy_name: [chunks]}
    """
    chunk_dict = {}
    for strategy in CHUNK_STRATEGIES.keys():
        if "fixed" in strategy:
            chunk_dict[strategy] = chunk_fixed(text, strategy)
        else:
            chunk_dict[strategy] = chunk_sentence(text)
    return chunk_dict