# app/streamlit_app.py
import sys
import os
import streamlit as st
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.cleaner import clean_pages
from src.ingestion.chunker import chunk_fixed
from src.retrieval.retriever import retrieve
from src.generation.generator import generate
from config import EMBEDDING_MODEL, VECTORSTORE_DIR
st.set_page_config(page_title="DocumentSync AI", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=VECTORSTORE_DIR)

def ingest_pdf(uploaded_file) -> int:
    """
    Parse, clean, chunk, embed uploaded PDF into
    ChromaDB collection 'uploaded_doc'.
    Returns: number of chunks stored
    """
    model = load_model()
    client = get_chroma_client()

    # delete existing collection to avoid stale chunks
    try:
        client.delete_collection("uploaded_doc")
    except:
        pass
    collection = client.get_or_create_collection(
        name="uploaded_doc",
        metadata={"hnsw:space": "cosine"}
    )

    # extract pages
    pages = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text)

    # clean and chunk
    cleaned_text = clean_pages(pages)
    chunks = chunk_fixed(text=cleaned_text, strategy_name="fixed_1024")
    embeddings = model.encode(chunks)

    # store each chunk
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"uploaded_{i}"],
            embeddings=[embeddings[i].tolist()],
            documents=[chunk],
            metadatas=[{"chunk_index": i}]
        )

    return len(chunks)

def main():
    st.title("DocumentSync AI")
    st.caption("Upload a PDF and ask questions about it")

    # initialise session state
    if "doc_ready" not in st.session_state:
        st.session_state["doc_ready"] = False
    if "chunk_count" not in st.session_state:
        st.session_state["chunk_count"] = 0

    # sidebar
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

        if uploaded_file:
            if uploaded_file.name != st.session_state.get("uploaded_filename"):
                with st.spinner("Processing document..."):
                    chunk_count = ingest_pdf(uploaded_file=uploaded_file)
                    st.session_state["chunk_count"] = chunk_count
                    st.session_state["doc_ready"] = True
                    st.session_state["uploaded_filename"] = uploaded_file.name

        if st.session_state["doc_ready"]:
            st.success(f"Ready! {st.session_state['chunk_count']} chunks indexed")

    # main area
    st.header("Ask a Question")
    query = st.text_input("Enter your question")

    if query and st.session_state["doc_ready"]:
        with st.spinner("Searching..."):
            retrieved_chunks = retrieve(
                query=query, 
                strategy_name="uploaded_doc",
                client=get_chroma_client()
            )
            response = generate(query=query, retrieved_chunks=retrieved_chunks)

        st.subheader("Answer")
        st.write(response["answer"])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latency", f"{response['latency_seconds']:.2f}s")
        with col2:
            st.metric("Chunks used", response["num_chunks_used"])

        st.subheader("Source Chunks")
        for i, chunk in enumerate(retrieved_chunks):
            with st.expander(f"Chunk {i+1} — Score: {chunk['score']:.3f}"):
                st.write(chunk["text"])

    elif query and not st.session_state["doc_ready"]:
        st.warning("Please upload a PDF first")

if __name__ == "__main__":
    main()