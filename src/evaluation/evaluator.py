# src/evaluation/evaluator.py
import json
import os
import random
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from src.retrieval.retriever import retrieve
from src.generation.generator import generate
from config import QA_PAIRS_DIR, RESULTS_DIR, GROQ_MODEL, EMBEDDING_MODEL, CHUNK_STRATEGIES

load_dotenv()
GROQ_JUDGE_API_KEY = os.getenv("GROQ_JUDGE_API_KEY")

SAMPLE_SIZE = 30
RESULTS_DIR = "eval/results"

def load_qa_pairs(strategy_name: str) -> list[dict]:
    """Load Q&A pairs from JSON file for a given strategy."""
    with open(os.path.join(QA_PAIRS_DIR, f"{strategy_name}_qa_pairs.json")) as f:
        qa_pairs = json.load(f)
    if len(qa_pairs) > SAMPLE_SIZE:
        qa_pairs = random.sample(qa_pairs, SAMPLE_SIZE)
    return qa_pairs

def run_rag_for_eval(qa_pairs: list[dict], strategy_name: str) -> dict:
    """
    Run retrieval + generation for each Q&A pair.
    Returns RAGAS-format dict with questions, answers, contexts, ground_truths
    """
    questions, answers, contexts, ground_truths = [], [], [], []
    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        ground_truth = qa["answer"]

        retrieved_chunks = retrieve(query=question, strategy_name=strategy_name)
        result = generate(query=question, retrieved_chunks=retrieved_chunks)

        questions.append(question)
        answers.append(result["answer"])
        contexts.append([chunk["text"] for chunk in retrieved_chunks])
        ground_truths.append(ground_truth)

        print(f" Ran {i+1}/{len(qa_pairs)} for {strategy_name}")
    
    return {
        "question" : questions,
        "answer" : answers,
        "contexts" : contexts,
        "ground_truth" : ground_truths
    }

def evaluate_strategy(strategy_name: str) -> pd.DataFrame:
    """
    Run full RAGAS eval for one strategy. Save results to CSV.
    Returns: dataframe of scores
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\nEvaluating strategy: {strategy_name}")
    qa_pairs = load_qa_pairs(strategy_name)
    ragas_input = run_rag_for_eval(qa_pairs, strategy_name)
    dataset = Dataset.from_dict(ragas_input)

    llm = LangchainLLMWrapper(ChatGroq(model=GROQ_MODEL, api_key=GROQ_JUDGE_API_KEY, temperature=0.1))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

    run_config = RunConfig(max_workers=1, timeout=180)

    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config
    )

    df = scores.to_pandas()
    df.to_csv(os.path.join(RESULTS_DIR, f"{strategy_name}_results.csv"), index=False)
    print(f"Saved results for {strategy_name}")
    return df

def evaluate_all():
    """Run eval for all 4 strategies and print comparison table."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {}
    for strategy_name in CHUNK_STRATEGIES.keys():
        df = evaluate_strategy(strategy_name)
        summary[strategy_name] = {
            "faithfulness" : df["faithfulness"].mean(),
            "answer_relevancy" : df["answer_relevancy"].mean(),
            "context_precision" : df["context_precision"].mean(),
            "context_recall" : df["context_recall"].mean()
        }
    
    print("\n=== RAGAS Evaluation Summary ===")
    summary_df = pd.DataFrame(summary).T
    print(summary_df.to_string())
    summary_df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"))

if __name__ == "__main__":
    evaluate_all()