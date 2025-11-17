# Test Results - Task A Retrieval System

## Test Run Summary

**Date:** Test run completed successfully  
**Environment:** mtrag virtual environment (Python 3.9.6)  
**Domain:** ClapNQ  
**Retriever:** BM25  
**Query Type:** Last Turn  
**Top-K:** 5

## Results

### Performance Metrics

| Metric | @1 | @3 | @5 |
|--------|----|----|----|
| **Recall** | 0.07088 | 0.1327 | 0.19297 |
| **nDCG** | 0.18269 | 0.15535 | 0.17804 |

### Comparison with Baseline

| Method | R@5 | nDCG@5 |
|--------|-----|--------|
| **Our BM25 (Last Turn)** | 0.193 | 0.178 |
| **Baseline BM25 (Last Turn)** | 0.20 | 0.18 |

✅ **Our results are very close to the baseline!**

## What Was Tested

1. ✅ Corpus loading from zip files
2. ✅ Query loading
3. ✅ BM25 indexing (183,408 documents)
4. ✅ Retrieval on 208 queries
5. ✅ Result formatting
6. ✅ Evaluation script integration

## Files Generated

- `results/test_bm25_clapnq.jsonl` - Retrieval results
- `results/test_bm25_clapnq_evaluated.jsonl` - Evaluated results with scores
- `results/test_bm25_clapnq_evaluated_aggregate.csv` - Aggregate metrics

## Next Steps

1. Try different retrievers (dense, hybrid)
2. Experiment with query types (questions, rewrite)
3. Tune BM25 parameters (k1, b)
4. Test on other domains (cloud, fiqa, govt)
5. Try different dense models
6. Experiment with hybrid fusion methods

## Command Used

```bash
cd experiments/retrieval
source ../../mtrag/bin/activate
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever bm25 \
  --query_type lastturn \
  --top_k 5 \
  --output_file results/test_bm25_clapnq.jsonl

# Evaluate
python ../../scripts/evaluation/run_retrieval_eval.py \
  --input_file results/test_bm25_clapnq.jsonl \
  --output_file results/test_bm25_clapnq_evaluated.jsonl
```

## System Status

✅ **All components working correctly!**

