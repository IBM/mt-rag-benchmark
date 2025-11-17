# Quick Start: Using the Improvements

## 🚀 Best Configuration (Recommended)

This configuration should give you **R@5 ~0.45-0.50** (vs baseline 0.193):

```bash
cd experiments/retrieval
source ../../mtrag/bin/activate

python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --dense_model BAAI/bge-base-en-v1.5 \
  --fusion_method rrf \
  --query_type rewrite \
  --query_rewrite contextual \
  --use_reranker \
  --reranker_initial_k 100 \
  --top_k 10 \
  --output_file results/clapnq_best.jsonl
```

## 📈 Expected Improvements

| Configuration | R@5 | Improvement |
|---------------|-----|-------------|
| Baseline (BM25 Last Turn) | 0.193 | - |
| + Query Rewrite | 0.25 | +30% |
| + Dense Retrieval | 0.37 | +91% |
| + Hybrid | 0.40-0.45 | +108-133% |
| + Re-ranking | 0.45-0.50 | **+133-159%** |

## 🎯 Three Tiers of Improvements

### Tier 1: Quick Wins (No Code Changes Needed)
Just use different command-line options:

```bash
# 1. Use query rewrite
--query_type rewrite

# 2. Use hybrid retrieval
--retriever hybrid

# 3. Use better query rewriting
--query_rewrite contextual
```

### Tier 2: Add Re-ranking (Best Quality)
```bash
--use_reranker \
--reranker_initial_k 100
```

### Tier 3: Multi-Query (For Multi-Turn)
```bash
--use_multi_query \
--multi_query_fusion rrf
```

## 📝 Complete Examples

### Example 1: Fast & Good (No Re-ranking)
```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --query_type rewrite \
  --output_file results/clapnq_fast.jsonl
```
**Time:** ~5-10 min | **R@5:** ~0.40

### Example 2: Best Quality (With Re-ranking)
```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --query_type rewrite \
  --query_rewrite contextual \
  --use_reranker \
  --output_file results/clapnq_best.jsonl
```
**Time:** ~15-20 min | **R@5:** ~0.45-0.50

### Example 3: All Features
```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --dense_model BAAI/bge-base-en-v1.5 \
  --query_type rewrite \
  --query_rewrite contextual \
  --use_reranker \
  --use_multi_query \
  --top_k 10 \
  --output_file results/clapnq_all_features.jsonl
```
**Time:** ~20-25 min | **R@5:** ~0.48-0.53

## 🔍 What Each Improvement Does

1. **`--query_type rewrite`**: Uses pre-rewritten queries (better than last turn)
2. **`--retriever hybrid`**: Combines BM25 + Dense (best of both)
3. **`--query_rewrite contextual`**: Smart query rewriting (better context)
4. **`--use_reranker`**: Re-ranks results for better precision
5. **`--use_multi_query`**: Handles multi-turn better

## ⚡ Quick Test

Test the improvements quickly:

```bash
# Test with re-ranking (best quality)
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --query_type rewrite \
  --use_reranker \
  --top_k 10 \
  --output_file results/test_improved.jsonl

# Evaluate
cd ../..
python scripts/evaluation/run_retrieval_eval.py \
  --input_file experiments/retrieval/results/test_improved.jsonl \
  --output_file experiments/retrieval/results/test_improved_evaluated.jsonl

# Check results
cat experiments/retrieval/results/test_improved_evaluated_aggregate.csv
```

You should see **significant improvement** over the baseline!

