#!/bin/bash
# Example script to run retrieval on all domains

# Make sure you're in the experiments/retrieval directory
cd "$(dirname "$0")"

# Create results directory
mkdir -p results

# Example 1: BM25 on ClapNQ
echo "Running BM25 on ClapNQ..."
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever bm25 \
  --query_type lastturn \
  --top_k 10 \
  --output_file results/clapnq_bm25_lastturn.jsonl

# Example 2: Dense retrieval on ClapNQ
echo "Running Dense retrieval on ClapNQ..."
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever dense \
  --dense_model BAAI/bge-base-en-v1.5 \
  --query_type lastturn \
  --top_k 10 \
  --output_file results/clapnq_dense_lastturn.jsonl

# Example 3: Hybrid retrieval on ClapNQ
echo "Running Hybrid retrieval on ClapNQ..."
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --dense_model BAAI/bge-base-en-v1.5 \
  --fusion_method rrf \
  --query_type lastturn \
  --top_k 10 \
  --output_file results/clapnq_hybrid_lastturn.jsonl

echo "Done! Results saved in results/ directory"

