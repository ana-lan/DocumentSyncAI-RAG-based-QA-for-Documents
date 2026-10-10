FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# 1. CPU-only torch first (skips ~4GB of CUDA libraries)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Python dependencies (cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3. Bake the embedding model and NLTK data into the image so pods start fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -m nltk.downloader punkt punkt_tab

# 4. App code last (changes most often)
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]