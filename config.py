# config.py — all tunable parameters in one place
import os
from dotenv import load_dotenv
load_dotenv()

# ── Corpus ────────────────────────────────────────────────
ARXIV_QUERY = "machine learning"       # change to target different domains
MAX_PAPERS   = 30                      # number of PDFs to download
RAW_DIR      = "data/raw"
PROCESSED_DIR = "data/processed"

# ── Text extraction ───────────────────────────────────────
# pdfplumber merges characters into one word when the gap between them is
# smaller than x_tolerance (points). Its default is 3. Week 2 picks a value
# with benchmarks/diagnose_extraction.py and sets it here.
PDF_X_TOLERANCE = float(os.getenv("PDF_X_TOLERANCE", "1.5"))
DROP_REFERENCES = os.getenv("DROP_REFERENCES", "1") == "1"   # cut the bibliography before chunking

# ── Chunking strategies ───────────────────────────────────
CHUNK_STRATEGIES = {
    "fixed_256":  {"chunk_size": 256,  "chunk_overlap": 32},
    "fixed_512":  {"chunk_size": 512,  "chunk_overlap": 64},
    "fixed_1024": {"chunk_size": 1024, "chunk_overlap": 128},
    "sentence":   {"chunk_size": None, "chunk_overlap": None},  # NLTK-based
}

# ── Embedding ─────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTORSTORE_DIR = "vectorstore"

# ── Retrieval ─────────────────────────────────────────────
TOP_K_VALUES         = [3, 5, 10]      # values to sweep during eval
DEFAULT_TOP_K        = 5
SIMILARITY_THRESHOLD = 0.2

# ── Generation ────────────────────────────────────────────
GROQ_MODEL  = "llama-3.1-8b-instant"
TOGETHER_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"  # kept so old imports don't break

# Backend switch: defaults point to Together, override via env vars for vLLM
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL    = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
LLM_API_KEY  = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "EMPTY")

MAX_TOKENS  = 512
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))   # low = more factual; evals use 0
PROMPT_STYLE = os.getenv("PROMPT_STYLE", "cite")   # "cite" numbers the chunks and asks for [n]; "plain" is week 1

# ── Evaluation ────────────────────────────────────────────
QA_PAIRS_DIR    = "eval/qa_pairs"
RESULTS_DIR     = "eval/results"
NUM_QA_PAIRS    = 200                  # total to generate
QA_GEN_MODEL = "llama-3.1-8b-instant"   # smarter model for Q&A gen