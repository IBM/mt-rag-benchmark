#!/bin/bash
# Script to run all improvement experiments

cd "$(dirname "$0")"
source ../../mtrag/bin/activate

mkdir -p results

DOMAIN="clapnq"  # Change to test other domains

echo "=========================================="
echo "Running Improvement Experiments"
echo "Domain: $DOMAIN"
echo "=========================================="

# Phase 1: Quick Wins
echo ""
echo "Phase 1: Quick Wins"
echo "-------------------"

echo "1. BM25 with Query Rewrite..."
python retrieval_pipeline.py \
  --domain $DOMAIN \
  --retriever bm25 \
  --query_type rewrite \
  --top_k 10 \
  --output_file results/${DOMAIN}_bm25_rewrite.jsonl

echo "2. BM25 with All Questions..."
python retrieval_pipeline.py \
  --domain $DOMAIN \
  --retriever bm25 \
  --query_type questions \
  --top_k 10 \
  --output_file results/${DOMAIN}_bm25_questions.jsonl

echo "3. Dense Retrieval (BGE-base)..."
python retrieval_pipeline.py \
  --domain $DOMAIN \
  --retriever dense \
  --dense_model BAAI/bge-base-en-v1.5 \
  --query_type lastturn \
  --top_k 10 \
  --output_file results/${DOMAIN}_dense_lastturn.jsonl

echo "4. Dense with Query Rewrite..."
python retrieval_pipeline.py \
  --domain $DOMAIN \
  --retriever dense \
  --dense_model BAAI/bge-base-en-v1.5 \
  --query_type rewrite \
  --top_k 10 \
  --output_file results/${DOMAIN}_dense_rewrite.jsonl

echo "5. Hybrid Retrieval (RRF)..."
python retrieval_pipeline.py \
  --domain $DOMAIN \
  --retriever hybrid \
  --dense_model BAAI/bge-base-en-v1.5 \
  --fusion_method rrf \
  --query_type rewrite \
  --top_k 10 \
  --output_file results/${DOMAIN}_hybrid_rrf_rewrite.jsonl

# Phase 2: Optimization
echo ""
echo "Phase 2: Optimization"
echo "----------------------"

echo "6. Hybrid with All Questions..."
python retrieval_pipeline.py \
  --domain $DOMAIN \
  --retriever hybrid \
  --dense_model BAAI/bge-base-en-v1.5 \
  --fusion_method rrf \
  --query_type questions \
  --top_k 10 \
  --output_file results/${DOMAIN}_hybrid_rrf_questions.jsonl

echo "7. Hybrid with Top-K=20..."
python retrieval_pipeline.py \
  --domain $DOMAIN \
  --retriever hybrid \
  --dense_model BAAI/bge-base-en-v1.5 \
  --fusion_method rrf \
  --query_type rewrite \
  --top_k 20 \
  --output_file results/${DOMAIN}_hybrid_rrf_rewrite_top20.jsonl

echo ""
echo "=========================================="
echo "Experiments Complete!"
echo "Now evaluating all results..."
echo "=========================================="

# Evaluate all results
cd ../..
for file in experiments/retrieval/results/${DOMAIN}_*.jsonl; do
  if [[ ! "$file" == *"_evaluated.jsonl" ]]; then
    output="${file%.jsonl}_evaluated.jsonl"
    echo "Evaluating $file..."
    python scripts/evaluation/run_retrieval_eval.py \
      --input_file "$file" \
      --output_file "$output" 2>&1 | tail -5
  fi
done

echo ""
echo "=========================================="
echo "All done! Check results in:"
echo "experiments/retrieval/results/${DOMAIN}_*_evaluated_aggregate.csv"
echo "=========================================="

