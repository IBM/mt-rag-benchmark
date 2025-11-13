#!/bin/bash
# Comprehensive test script for all improvements

set -e  # Exit on error

cd "$(dirname "$0")"
source ../../mtrag/bin/activate

echo "=========================================="
echo "Comprehensive Test - All Improvements"
echo "=========================================="
echo ""

# Create results directory
mkdir -p results

DOMAIN="clapnq"  # Change to test other domains

# Test configurations
declare -a tests=(
    "bm25:lastturn:BM25 Baseline"
    "bm25:rewrite:BM25 + Rewrite"
    "dense:rewrite:Dense + Rewrite"
    "hybrid:rewrite:Hybrid + Rewrite"
    "hybrid:rewrite:contextual:Hybrid + Contextual Rewrite"
    "hybrid:rewrite::reranker:Hybrid + Re-ranking"
)

for test_config in "${tests[@]}"; do
    IFS=':' read -r retriever query_type query_rewrite reranker_flag test_name <<< "$test_config"
    
    echo "Testing: $test_name"
    
    # Build command
    cmd="python retrieval_pipeline.py --domain $DOMAIN --retriever $retriever --query_type $query_type --top_k 10"
    
    # Add dense model for dense/hybrid
    if [ "$retriever" == "dense" ] || [ "$retriever" == "hybrid" ]; then
        cmd="$cmd --dense_model BAAI/bge-base-en-v1.5"
    fi
    
    # Add query rewrite if specified
    if [ -n "$query_rewrite" ]; then
        cmd="$cmd --query_rewrite $query_rewrite"
    fi
    
    # Add re-ranker if specified
    if [ -n "$reranker_flag" ]; then
        cmd="$cmd --use_reranker"
    fi
    
    # Generate output filename
    filename=$(echo "$test_name" | tr ' ' '_' | tr '+' '_' | tr ':' '_' | tr '[:upper:]' '[:lower:]')
    cmd="$cmd --output_file results/test_${filename}.jsonl"
    
    # Run test
    eval $cmd
    
    echo "✓ $test_name complete"
    echo ""
done

# Evaluate all
echo "Evaluating all results..."
cd ../..
for file in experiments/retrieval/results/test_*.jsonl; do
    if [[ ! "$file" == *"_evaluated.jsonl" ]]; then
        output="${file%.jsonl}_evaluated.jsonl"
        echo "Evaluating $(basename $file)..."
        python scripts/evaluation/run_retrieval_eval.py \
            --input_file "$file" \
            --output_file "$output" 2>&1 | tail -3
    fi
done

echo ""
echo "=========================================="
echo "All Results Summary"
echo "=========================================="
echo ""
for file in experiments/retrieval/results/test_*_evaluated_aggregate.csv; do
    if [ -f "$file" ]; then
        echo "File: $(basename $file)"
        cat "$file"
        echo ""
    fi
done

echo "=========================================="
echo "Test Complete!"
echo "=========================================="

