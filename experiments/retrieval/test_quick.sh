#!/bin/bash
# Quick test script for retrieval system

set -e  # Exit on error

cd "$(dirname "$0")"
source ../../mtrag/bin/activate

echo "=========================================="
echo "Quick Test - Task A Retrieval System"
echo "=========================================="
echo ""

# Create results directory
mkdir -p results

# Test 1: BM25 Baseline
echo "Test 1: BM25 Baseline (Last Turn)..."
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever bm25 \
  --query_type lastturn \
  --top_k 10 \
  --output_file results/test_bm25_baseline.jsonl

echo "✓ BM25 test complete"
echo ""

# Test 2: Hybrid with Rewrite (Quick Improvement)
echo "Test 2: Hybrid + Rewrite..."
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --dense_model BAAI/bge-base-en-v1.5 \
  --query_type rewrite \
  --top_k 10 \
  --output_file results/test_hybrid_rewrite.jsonl

echo "✓ Hybrid test complete"
echo ""

# Evaluate both
echo "Evaluating results..."
cd ../..
python scripts/evaluation/run_retrieval_eval.py \
  --input_file experiments/retrieval/results/test_bm25_baseline.jsonl \
  --output_file experiments/retrieval/results/test_bm25_baseline_evaluated.jsonl

python scripts/evaluation/run_retrieval_eval.py \
  --input_file experiments/retrieval/results/test_hybrid_rewrite.jsonl \
  --output_file experiments/retrieval/results/test_hybrid_rewrite_evaluated.jsonl

echo ""
echo "=========================================="
echo "Results Comparison"
echo "=========================================="
echo ""
echo "BM25 Baseline:"
cat experiments/retrieval/results/test_bm25_baseline_evaluated_aggregate.csv
echo ""
echo "Hybrid + Rewrite:"
cat experiments/retrieval/results/test_hybrid_rewrite_evaluated_aggregate.csv
echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="

