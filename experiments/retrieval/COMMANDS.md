# Task A: Complete Command Reference

This document lists all commands needed to run Task A (Retrieval Only) experiments.

## Table of Contents
1. [Setup Commands](#setup-commands)
2. [BM25 Retrieval](#bm25-retrieval)
3. [Dense Retrieval](#dense-retrieval)
4. [Hybrid Retrieval](#hybrid-retrieval)
5. [Evaluation Commands](#evaluation-commands)
6. [Batch Processing](#batch-processing)
7. [Advanced Options](#advanced-options)

---

## Setup Commands

### Activate Virtual Environment
```bash
cd /Users/pratibharevankar/Desktop/mt-rag-benchmark
source mtrag/bin/activate
```

### Install Dependencies (if needed)
```bash
cd experiments/retrieval
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Verify Installation
```bash
python -c "import rank_bm25; import sentence_transformers; import nltk; print('All dependencies OK')"
```

---

## BM25 Retrieval

### Basic BM25 on Single Domain
```bash
cd experiments/retrieval
source ../../mtrag/bin/activate

python retrieval_pipeline.py \
  --domain clapnq \
  --retriever bm25 \
  --query_type lastturn \
  --top_k 10 \
  --output_file results/clapnq_bm25_lastturn.jsonl
```

### BM25 with Custom Parameters
```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever bm25 \
  --bm25_k1 1.2 \
  --bm25_b 0.8 \
  --query_type lastturn \
  --top_k 10 \
  --output_file results/clapnq_bm25_custom.jsonl
```

### BM25 on All Domains
```bash
# ClapNQ
python retrieval_pipeline.py --domain clapnq --retriever bm25 --query_type lastturn --top_k 10 --output_file results/clapnq_bm25.jsonl

# Cloud
python retrieval_pipeline.py --domain cloud --retriever bm25 --query_type lastturn --top_k 10 --output_file results/cloud_bm25.jsonl

# FiQA
python retrieval_pipeline.py --domain fiqa --retriever bm25 --query_type lastturn --top_k 10 --output_file results/fiqa_bm25.jsonl

# Govt
python retrieval_pipeline.py --domain govt --retriever bm25 --query_type lastturn --top_k 10 --output_file results/govt_bm25.jsonl
```

### BM25 with Different Query Types
```bash
# Last turn only
python retrieval_pipeline.py --domain clapnq --retriever bm25 --query_type lastturn --output_file results/clapnq_bm25_lastturn.jsonl

# All questions
python retrieval_pipeline.py --domain clapnq --retriever bm25 --query_type questions --output_file results/clapnq_bm25_questions.jsonl

# Rewritten queries
python retrieval_pipeline.py --domain clapnq --retriever bm25 --query_type rewrite --output_file results/clapnq_bm25_rewrite.jsonl
```

---

## Dense Retrieval

### Basic Dense Retrieval (BGE-base)
```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever dense \
  --dense_model BAAI/bge-base-en-v1.5 \
  --query_type lastturn \
  --top_k 10 \
  --output_file results/clapnq_dense_bge_base.jsonl
```

### Dense Retrieval with Different Models
```bash
# BGE-base
python retrieval_pipeline.py --domain clapnq --retriever dense --dense_model BAAI/bge-base-en-v1.5 --output_file results/clapnq_dense_bge_base.jsonl

# BGE-large
python retrieval_pipeline.py --domain clapnq --retriever dense --dense_model BAAI/bge-large-en-v1.5 --output_file results/clapnq_dense_bge_large.jsonl

# E5-base
python retrieval_pipeline.py --domain clapnq --retriever dense --dense_model intfloat/e5-base-v2 --output_file results/clapnq_dense_e5_base.jsonl

# E5-large
python retrieval_pipeline.py --domain clapnq --retriever dense --dense_model intfloat/e5-large-v2 --output_file results/clapnq_dense_e5_large.jsonl
```

### Dense Retrieval with GPU
```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever dense \
  --dense_model BAAI/bge-base-en-v1.5 \
  --device cuda \
  --batch_size 64 \
  --output_file results/clapnq_dense_gpu.jsonl
```

### Dense Retrieval on All Domains
```bash
for domain in clapnq cloud fiqa govt; do
  python retrieval_pipeline.py \
    --domain $domain \
    --retriever dense \
    --dense_model BAAI/bge-base-en-v1.5 \
    --query_type lastturn \
    --top_k 10 \
    --output_file results/${domain}_dense.jsonl
done
```

---

## Hybrid Retrieval

### Hybrid with RRF (Reciprocal Rank Fusion) - Recommended
```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --dense_model BAAI/bge-base-en-v1.5 \
  --fusion_method rrf \
  --query_type lastturn \
  --top_k 10 \
  --output_file results/clapnq_hybrid_rrf.jsonl
```

### Hybrid with Weighted Fusion
```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --dense_model BAAI/bge-base-en-v1.5 \
  --fusion_method weighted \
  --alpha 0.6 \
  --query_type lastturn \
  --top_k 10 \
  --output_file results/clapnq_hybrid_weighted.jsonl
```

### Hybrid on All Domains
```bash
for domain in clapnq cloud fiqa govt; do
  python retrieval_pipeline.py \
    --domain $domain \
    --retriever hybrid \
    --dense_model BAAI/bge-base-en-v1.5 \
    --fusion_method rrf \
    --query_type lastturn \
    --top_k 10 \
    --output_file results/${domain}_hybrid.jsonl
done
```

---

## Evaluation Commands

### Evaluate Single Result File
```bash
cd /Users/pratibharevankar/Desktop/mt-rag-benchmark
source mtrag/bin/activate

python scripts/evaluation/run_retrieval_eval.py \
  --input_file experiments/retrieval/results/clapnq_bm25.jsonl \
  --output_file experiments/retrieval/results/clapnq_bm25_evaluated.jsonl
```

### Evaluate All BM25 Results
```bash
for domain in clapnq cloud fiqa govt; do
  python scripts/evaluation/run_retrieval_eval.py \
    --input_file experiments/retrieval/results/${domain}_bm25.jsonl \
    --output_file experiments/retrieval/results/${domain}_bm25_evaluated.jsonl
done
```

### Evaluate All Dense Results
```bash
for domain in clapnq cloud fiqa govt; do
  python scripts/evaluation/run_retrieval_eval.py \
    --input_file experiments/retrieval/results/${domain}_dense.jsonl \
    --output_file experiments/retrieval/results/${domain}_dense_evaluated.jsonl
done
```

### Evaluate All Hybrid Results
```bash
for domain in clapnq cloud fiqa govt; do
  python scripts/evaluation/run_retrieval_eval.py \
    --input_file experiments/retrieval/results/${domain}_hybrid.jsonl \
    --output_file experiments/retrieval/results/${domain}_hybrid_evaluated.jsonl
done
```

### View Aggregate Results
```bash
# View CSV files with aggregate metrics
cat experiments/retrieval/results/*_evaluated_aggregate.csv
```

---

## Batch Processing

### Run All Retrievers on All Domains
```bash
cd experiments/retrieval
source ../../mtrag/bin/activate

# Create results directory
mkdir -p results

# BM25 on all domains
for domain in clapnq cloud fiqa govt; do
  echo "Running BM25 on $domain..."
  python retrieval_pipeline.py \
    --domain $domain \
    --retriever bm25 \
    --query_type lastturn \
    --top_k 10 \
    --output_file results/${domain}_bm25.jsonl
done

# Dense on all domains
for domain in clapnq cloud fiqa govt; do
  echo "Running Dense on $domain..."
  python retrieval_pipeline.py \
    --domain $domain \
    --retriever dense \
    --dense_model BAAI/bge-base-en-v1.5 \
    --query_type lastturn \
    --top_k 10 \
    --output_file results/${domain}_dense.jsonl
done

# Hybrid on all domains
for domain in clapnq cloud fiqa govt; do
  echo "Running Hybrid on $domain..."
  python retrieval_pipeline.py \
    --domain $domain \
    --retriever hybrid \
    --dense_model BAAI/bge-base-en-v1.5 \
    --fusion_method rrf \
    --query_type lastturn \
    --top_k 10 \
    --output_file results/${domain}_hybrid.jsonl
done
```

### Evaluate All Results
```bash
cd /Users/pratibharevankar/Desktop/mt-rag-benchmark
source mtrag/bin/activate

for file in experiments/retrieval/results/*.jsonl; do
  if [[ ! "$file" == *"_evaluated.jsonl" ]]; then
    output="${file%.jsonl}_evaluated.jsonl"
    echo "Evaluating $file..."
    python scripts/evaluation/run_retrieval_eval.py \
      --input_file "$file" \
      --output_file "$output"
  fi
done
```

---

## Advanced Options

### Query Rewriting
```bash
# Last turn only
python retrieval_pipeline.py --domain clapnq --retriever hybrid \
  --query_rewrite last_turn --output_file results/clapnq_lastturn.jsonl

# All turns
python retrieval_pipeline.py --domain clapnq --retriever hybrid \
  --query_rewrite all_turns --output_file results/clapnq_allturns.jsonl

# Expanded query
python retrieval_pipeline.py --domain clapnq --retriever hybrid \
  --query_rewrite expand --output_file results/clapnq_expanded.jsonl

# Full conversation
python retrieval_pipeline.py --domain clapnq --retriever hybrid \
  --query_rewrite full --output_file results/clapnq_full.jsonl
```

### Custom Corpus/Query Paths
```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --corpus_path /path/to/custom/corpus.jsonl \
  --queries_path /path/to/custom/queries.jsonl \
  --retriever bm25 \
  --output_file results/custom.jsonl
```

### Different Top-K Values
```bash
# Top 5
python retrieval_pipeline.py --domain clapnq --retriever hybrid --top_k 5 --output_file results/clapnq_top5.jsonl

# Top 10
python retrieval_pipeline.py --domain clapnq --retriever hybrid --top_k 10 --output_file results/clapnq_top10.jsonl

# Top 20
python retrieval_pipeline.py --domain clapnq --retriever hybrid --top_k 20 --output_file results/clapnq_top20.jsonl
```

### CPU vs GPU
```bash
# Force CPU
python retrieval_pipeline.py --domain clapnq --retriever dense --device cpu --output_file results/clapnq_cpu.jsonl

# Use GPU (if available)
python retrieval_pipeline.py --domain clapnq --retriever dense --device cuda --output_file results/clapnq_gpu.jsonl
```

### Batch Size Tuning
```bash
# Small batch (for limited memory)
python retrieval_pipeline.py --domain clapnq --retriever dense --batch_size 16 --output_file results/clapnq_smallbatch.jsonl

# Large batch (for faster processing)
python retrieval_pipeline.py --domain clapnq --retriever dense --batch_size 64 --output_file results/clapnq_largebatch.jsonl
```

---

## Quick Reference: Complete Workflow

### 1. Setup (One-time)
```bash
cd /Users/pratibharevankar/Desktop/mt-rag-benchmark
source mtrag/bin/activate
cd experiments/retrieval
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Run Retrieval (Example: Hybrid on ClapNQ)
```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --dense_model BAAI/bge-base-en-v1.5 \
  --fusion_method rrf \
  --query_type lastturn \
  --top_k 10 \
  --output_file results/clapnq_hybrid.jsonl
```

### 3. Evaluate Results
```bash
cd /Users/pratibharevankar/Desktop/mt-rag-benchmark
source mtrag/bin/activate

python scripts/evaluation/run_retrieval_eval.py \
  --input_file experiments/retrieval/results/clapnq_hybrid.jsonl \
  --output_file experiments/retrieval/results/clapnq_hybrid_evaluated.jsonl
```

### 4. View Results
```bash
# View aggregate metrics
cat experiments/retrieval/results/clapnq_hybrid_evaluated_aggregate.csv

# View individual query scores (first few lines)
head -5 experiments/retrieval/results/clapnq_hybrid_evaluated.jsonl | python -m json.tool
```

---

## Command Line Arguments Reference

### Required Arguments
- `--domain`: Domain name (clapnq, cloud, fiqa, govt)
- `--retriever`: Retriever type (bm25, dense, hybrid)
- `--output_file`: Output file path

### Optional Arguments

**Data:**
- `--corpus_path`: Custom corpus path (default: auto-detect)
- `--queries_path`: Custom queries path (default: auto-detect)
- `--query_type`: Query type (lastturn, questions, rewrite)

**Retriever Configuration:**
- `--dense_model`: Dense model name (default: BAAI/bge-base-en-v1.5)
- `--bm25_k1`: BM25 parameter k1 (default: 1.5)
- `--bm25_b`: BM25 parameter b (default: 0.75)
- `--alpha`: Weight for dense in hybrid (default: 0.5)
- `--fusion_method`: Fusion method (rrf, weighted)

**Query Processing:**
- `--query_rewrite`: Query rewriting (last_turn, all_turns, expand, full)

**Retrieval:**
- `--top_k`: Number of documents to retrieve (default: 10)

**Other:**
- `--device`: Device (cuda/cpu, default: auto)
- `--batch_size`: Batch size for dense (default: 32)

---

## Tips

1. **Start with BM25**: Fastest, good baseline
2. **Try Hybrid**: Usually best performance (combines BM25 + Dense)
3. **Experiment with Query Types**: `questions` often better than `lastturn` for multi-turn
4. **Use GPU for Dense**: Much faster if available
5. **Tune Top-K**: Start with 10, increase if needed
6. **Evaluate Everything**: Always run evaluation to compare methods

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch_size 16

# Use smaller model
--dense_model BAAI/bge-base-en-v1.5  # instead of large
```

### Slow Processing
```bash
# Use GPU
--device cuda

# Increase batch size (if memory allows)
--batch_size 64
```

### Missing Files
```bash
# Check if corpora exist
ls corpora/passage_level/*.zip

# Check if queries exist
ls human/retrieval_tasks/*/qrels/dev.tsv
```

