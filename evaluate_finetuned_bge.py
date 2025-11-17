import pathlib
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch 
import torch
import numpy as np

# --- 1. Setup Device and Model ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using NVIDIA GPU (cuda) for acceleration.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple MPS (M-series GPU) for acceleration.")
else:
    device = torch.device("cpu")
    print("⚠️ No GPU found. Using CPU (this will be very slow).")

MODEL_NAME = "./bge-finetuned-all-domains" 
print(f"Loading model: {MODEL_NAME} (this is your fine-tuned model)...")

model = SentenceBERT(MODEL_NAME, device=device.type)

# --- 2. Setup Retriever and Evaluator ---
retriever = DenseRetrievalExactSearch(model, batch_size=128)
k_values = [5, 10]
evaluator = EvaluateRetrieval(retriever, k_values=k_values)

# --- 3. Define All Domains ---
domains = ["clapnq", "fiqa", "govt", "cloud"]
all_results = {}

print("\n--- Running Full FINE-TUNED BGE Baseline Replication ---")

for domain in domains:
    print(f"\n--- Processing Domain: {domain} ---")
    
    # --- 4. Define Paths ---
    data_root = pathlib.Path(".")
    corpus_path = data_root / "corpora" / "passage_level"
    
    corpus_file = corpus_path / f"{domain}.jsonl"
    query_path = data_root / "human" / "retrieval_tasks" / domain
    query_file = query_path / f"{domain}_questions.jsonl" 
    qrels_file = query_path / "qrels" / "dev.tsv"

    print(f"Loading data: {domain}...")
    try:
        corpus, queries, qrels = GenericDataLoader(
            corpus_file=str(corpus_file),
            query_file=str(query_file),
            qrels_file=str(qrels_file)
        ).load_custom()
    except Exception as e:
        print(f"Error loading data for {domain}: {e}")
        continue

    print(f"Corpus: {len(corpus)} docs | Queries: {len(queries)} queries")

    # --- 5. Run Retrieval ---
    print(f"Running retrieval for {domain} (This will take a while)...")
    results = evaluator.retrieve(corpus, queries)

    # --- 6. Evaluate and Store ---
    print("Evaluating results...")
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
    
    all_results[domain] = {
        "Recall@5": recall['Recall@5'],
        "Recall@10": recall['Recall@10'],
        "nDCG@5": ndcg['NDCG@5'],
        "nDCG@10": ndcg['NDCG@10']
    }
    
    print(f"--- Results for {domain} ---")
    print(f"Recall@10: {recall['Recall@10']:.4f}")
    print(f"nDCG@10:   {ndcg['NDCG@10']:.4f}")

# --- 7. Calculate and Print Final Average ---
print("\n\n--- Finished All Domains: Final Summary ---")

avg_recall_5 = np.mean([res["Recall@5"] for res in all_results.values()])
avg_recall_10 = np.mean([res["Recall@10"] for res in all_results.values()])
avg_ndcg_5 = np.mean([res["nDCG@5"] for res in all_results.values()])
avg_ndcg_10 = np.mean([res["nDCG@10"] for res in all_results.values()])

# Get proposal numbers from Katsis et al. (2025) Table 3 (BGE-base 1.5, Last Turn)
paper_recall_5 = 0.30
paper_recall_10 = 0.38
paper_ndcg_5 = 0.27
paper_ndcg_10 = 0.30

print("--- Your Fine-Tuned BGE vs. Pre-Trained BGE (Paper) ---")
print(f"{'Metric':<12} | {'Your Avg.':<10} | {'Pre-Trained BGE (Paper)':<25}")
print("-" * 55)
print(f"{'Recall@5':<12} | {avg_recall_5:<10.4f} | {paper_recall_5:<25.4f}")
print(f"{'Recall@10':<12} | {avg_recall_10:<10.4f} | {paper_recall_10:<25.4f}")
print(f"{'nDCG@5':<12} | {avg_ndcg_5:<10.4f} | {paper_ndcg_5:<25.4f}")
print(f"{'nDCG@10':<12} | {avg_ndcg_10:<10.4f} | {paper_ndcg_10:<25.4f}")

print("\n--- Individual Domain Scores (Your Fine-Tuned Model) ---")
print(f"{'Domain':<10} | {'R@10':<7} | {'nDCG@10':<7}")
print("-" * 28)
for domain, scores in all_results.items():
    print(f"{domain:<10} | {scores['Recall@10']:<7.4f} | {scores['nDCG@10']:<7.4f}")

print("\nFine-tuned BGE evaluation complete.")