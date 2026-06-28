# config.py — all tunable parameters in one place

# ── Corpus ────────────────────────────────────────────────
ARXIV_QUERY = "machine learning"       # change to target different domains
MAX_PAPERS   = 30                      # number of PDFs to download
RAW_DIR      = "data/raw"
PROCESSED_DIR = "data/processed"

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
SIMILARITY_THRESHOLD = 0.3

# ── Generation ────────────────────────────────────────────
GROQ_MODEL  = "llama-3.1-8b-instant"
MAX_TOKENS  = 512
TEMPERATURE = 0.2                      # low = more factual

# ── Evaluation ────────────────────────────────────────────
QA_PAIRS_DIR    = "eval/qa_pairs"
RESULTS_DIR     = "eval/results"
NUM_QA_PAIRS    = 200                  # total to generate
QA_GEN_MODEL    = "llama-3.1-70b-versatile"   # smarter model for Q&A gen