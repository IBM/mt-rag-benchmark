# How to Improve Retrieval Scores

## Current Performance vs Baselines

| Method | R@5 | nDCG@5 | Improvement Potential |
|--------|-----|--------|----------------------|
| **Your BM25 (Last Turn)** | 0.193 | 0.178 | Baseline |
| BM25 Query Rewrite | 0.25 | 0.22 | **+30%** |
| BGE-base 1.5 (Last Turn) | 0.30 | 0.27 | **+55%** |
| BGE-base 1.5 Query Rewrite | 0.37 | 0.34 | **+91%** |
| Elser (Last Turn) | 0.49 | 0.45 | **+154%** |
| Elser Query Rewrite | 0.52 | 0.48 | **+170%** |

## 🎯 Quick Wins (Easy Improvements)

### 1. Use Query Rewrite Instead of Last Turn
**Expected Improvement: +30% (R@5: 0.193 → 0.25)**

```bash
# Instead of lastturn, use rewrite queries
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever bm25 \
  --query_type rewrite \
  --top_k 10 \
  --output_file results/clapnq_bm25_rewrite.jsonl
```

### 2. Use "Questions" Query Type (All User Questions)
**Expected Improvement: Better context understanding**

```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever bm25 \
  --query_type questions \
  --top_k 10 \
  --output_file results/clapnq_bm25_questions.jsonl
```

### 3. Switch to Dense Retrieval
**Expected Improvement: +55% (R@5: 0.193 → 0.30)**

```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever dense \
  --dense_model BAAI/bge-base-en-v1.5 \
  --query_type lastturn \
  --top_k 10 \
  --output_file results/clapnq_dense.jsonl
```

### 4. Combine Dense + Query Rewrite
**Expected Improvement: +91% (R@5: 0.193 → 0.37)**

```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever dense \
  --dense_model BAAI/bge-base-en-v1.5 \
  --query_type rewrite \
  --top_k 10 \
  --output_file results/clapnq_dense_rewrite.jsonl
```

### 5. Use Hybrid Retrieval (BM25 + Dense)
**Expected Improvement: Best of both worlds**

```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --dense_model BAAI/bge-base-en-v1.5 \
  --fusion_method rrf \
  --query_type rewrite \
  --top_k 10 \
  --output_file results/clapnq_hybrid_rewrite.jsonl
```

---

## 🚀 Advanced Improvements

### 6. Use Better Dense Models

**BGE-Large (Better but slower):**
```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever dense \
  --dense_model BAAI/bge-large-en-v1.5 \
  --query_type rewrite \
  --output_file results/clapnq_bge_large.jsonl
```

**E5 Models:**
```bash
# E5-base
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever dense \
  --dense_model intfloat/e5-base-v2 \
  --query_type rewrite \
  --output_file results/clapnq_e5_base.jsonl

# E5-large
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever dense \
  --dense_model intfloat/e5-large-v2 \
  --query_type rewrite \
  --output_file results/clapnq_e5_large.jsonl
```

### 7. Tune BM25 Parameters

**Experiment with different k1 and b values:**
```bash
# More term frequency weight
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever bm25 \
  --bm25_k1 2.0 \
  --bm25_b 0.75 \
  --query_type rewrite \
  --output_file results/clapnq_bm25_k1_2.0.jsonl

# More length normalization
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever bm25 \
  --bm25_k1 1.5 \
  --bm25_b 0.9 \
  --query_type rewrite \
  --output_file results/clapnq_bm25_b_0.9.jsonl
```

### 8. Increase Top-K for Better Recall

**Retrieve more documents (helps with recall):**
```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --top_k 20 \
  --query_type rewrite \
  --output_file results/clapnq_hybrid_top20.jsonl
```

### 9. Tune Hybrid Fusion

**Try different fusion methods and weights:**
```bash
# Weighted fusion with more weight on dense
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --fusion_method weighted \
  --alpha 0.7 \
  --query_type rewrite \
  --output_file results/clapnq_hybrid_weighted_0.7.jsonl

# RRF (usually better)
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --fusion_method rrf \
  --query_type rewrite \
  --output_file results/clapnq_hybrid_rrf.jsonl
```

---

## 🔬 Experimental Improvements (Require Code Changes)

### 10. Add Re-ranking

Re-ranking with a cross-encoder can significantly improve results:
- Retrieve top 100 with BM25/Dense
- Re-rank top 100 → top 10 with cross-encoder
- Expected improvement: +10-20%

### 11. Query Expansion

Expand queries with synonyms, related terms, or LLM-generated expansions:
- Use WordNet for synonyms
- Use LLM to expand queries
- Expected improvement: +5-15%

### 12. Multi-Query Retrieval

For multi-turn conversations:
- Retrieve separately for each turn
- Merge and deduplicate results
- Expected improvement: +5-10%

### 13. Domain-Specific Tuning

Tune parameters per domain:
- Different BM25 parameters for each domain
- Different dense models for each domain
- Expected improvement: +5-10%

---

## 📊 Recommended Experiment Order

### Phase 1: Quick Wins (Do These First)
1. ✅ Use `query_type rewrite` instead of `lastturn`
2. ✅ Try `query_type questions` 
3. ✅ Switch to dense retrieval (BGE-base)
4. ✅ Combine dense + rewrite

### Phase 2: Optimization
5. ✅ Try hybrid retrieval (RRF fusion)
6. ✅ Experiment with larger models (BGE-large, E5)
7. ✅ Tune BM25 parameters
8. ✅ Increase top_k to 20

### Phase 3: Advanced
9. Add re-ranking
10. Query expansion
11. Multi-query retrieval
12. Domain-specific tuning

---

## 🎯 Expected Results After Improvements

| Strategy | Expected R@5 | Expected nDCG@5 |
|----------|--------------|-----------------|
| Current (BM25 Last Turn) | 0.193 | 0.178 |
| BM25 + Rewrite | 0.25 | 0.22 |
| Dense (BGE-base) + Rewrite | 0.37 | 0.34 |
| Hybrid + Rewrite | 0.40-0.45 | 0.35-0.40 |
| Hybrid + Rewrite + Re-rank | 0.45-0.50 | 0.40-0.45 |

---

## 🧪 Testing Strategy

### 1. Create Test Script
```bash
# Test all combinations
for retriever in bm25 dense hybrid; do
  for query_type in lastturn questions rewrite; do
    python retrieval_pipeline.py \
      --domain clapnq \
      --retriever $retriever \
      --query_type $query_type \
      --output_file results/clapnq_${retriever}_${query_type}.jsonl
  done
done
```

### 2. Evaluate All
```bash
for file in results/clapnq_*.jsonl; do
  python ../../scripts/evaluation/run_retrieval_eval.py \
    --input_file $file \
    --output_file ${file%.jsonl}_evaluated.jsonl
done
```

### 3. Compare Results
```bash
# View all aggregate results
cat results/*_evaluated_aggregate.csv | grep -E "(collection|clapnq)"
```

---

## 💡 Key Insights from Baselines

1. **Query Rewrite is Critical**: Always improves results (+25-30%)
2. **Dense > Lexical**: BGE-base beats BM25 significantly
3. **Hybrid is Best**: Combines strengths of both
4. **Elser is Top Performer**: But requires Elasticsearch setup

---

## 🎓 Best Practices

1. **Always use query rewrite** - Biggest single improvement
2. **Start with hybrid** - Best balance of performance and complexity
3. **Use GPU for dense** - Much faster
4. **Experiment systematically** - Test one variable at a time
5. **Evaluate everything** - Track what works
6. **Domain matters** - Different domains may need different approaches

---

## 📈 Monitoring Progress

Track your improvements:
```bash
# Create comparison table
echo "Method,R@5,nDCG@5" > results/comparison.csv
grep "clapnq" results/*_evaluated_aggregate.csv | \
  awk -F',' '{print $4","$2","$3}' >> results/comparison.csv
```

---

## 🚀 Next Steps

1. **Run Phase 1 improvements** (should get you to ~0.37 R@5)
2. **Compare results** and identify best combination
3. **Apply to all domains** (clapnq, cloud, fiqa, govt)
4. **Consider advanced techniques** if needed
5. **Prepare for test phase** with best configuration

Good luck! 🎯

