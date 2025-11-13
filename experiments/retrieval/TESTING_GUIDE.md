# Testing Guide for Task A Retrieval System

## 🧪 Quick Test (5 minutes)

### Step 1: Activate Environment
```bash
cd /Users/pratibharevankar/Desktop/mt-rag-benchmark
source mtrag/bin/activate
cd experiments/retrieval
```

### Step 2: Run Basic Test (BM25)
```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever bm25 \
  --query_type lastturn \
  --top_k 10 \
  --output_file results/test_bm25.jsonl
```

### Step 3: Evaluate
```bash
cd ../..
python scripts/evaluation/run_retrieval_eval.py \
  --input_file experiments/retrieval/results/test_bm25.jsonl \
  --output_file experiments/retrieval/results/test_bm25_evaluated.jsonl
```

### Step 4: View Results
```bash
cat experiments/retrieval/results/test_bm25_evaluated_aggregate.csv
```

---

## 🚀 Test Improvements (Recommended)

### Test 1: Hybrid Retrieval (Quick Improvement)
```bash
cd experiments/retrieval
source ../../mtrag/bin/activate

python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --dense_model BAAI/bge-base-en-v1.5 \
  --query_type rewrite \
  --top_k 10 \
  --output_file results/test_hybrid.jsonl

# Evaluate
cd ../..
python scripts/evaluation/run_retrieval_eval.py \
  --input_file experiments/retrieval/results/test_hybrid.jsonl \
  --output_file experiments/retrieval/results/test_hybrid_evaluated.jsonl

# Compare
echo "=== BM25 Results ==="
cat experiments/retrieval/results/test_bm25_evaluated_aggregate.csv
echo ""
echo "=== Hybrid Results ==="
cat experiments/retrieval/results/test_hybrid_evaluated_aggregate.csv
```

**Expected:** Hybrid should show ~2x improvement over BM25

### Test 2: With Re-ranking (Best Quality)
```bash
cd experiments/retrieval
source ../../mtrag/bin/activate

python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --dense_model BAAI/bge-base-en-v1.5 \
  --query_type rewrite \
  --query_rewrite contextual \
  --use_reranker \
  --reranker_initial_k 100 \
  --top_k 10 \
  --output_file results/test_reranked.jsonl

# Evaluate
cd ../..
python scripts/evaluation/run_retrieval_eval.py \
  --input_file experiments/retrieval/results/test_reranked.jsonl \
  --output_file experiments/retrieval/results/test_reranked_evaluated.jsonl

# View results
cat experiments/retrieval/results/test_reranked_evaluated_aggregate.csv
```

**Expected:** Should show best results (R@5 ~0.45-0.50)

---

## 📊 Comprehensive Testing

### Test All Configurations

Create a test script:

```bash
cd experiments/retrieval
source ../../mtrag/bin/activate

# Create results directory
mkdir -p results

# Test 1: BM25 Baseline
echo "Test 1: BM25 Baseline..."
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever bm25 \
  --query_type lastturn \
  --top_k 10 \
  --output_file results/test1_bm25_baseline.jsonl

# Test 2: BM25 + Rewrite
echo "Test 2: BM25 + Rewrite..."
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever bm25 \
  --query_type rewrite \
  --top_k 10 \
  --output_file results/test2_bm25_rewrite.jsonl

# Test 3: Dense
echo "Test 3: Dense Retrieval..."
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever dense \
  --dense_model BAAI/bge-base-en-v1.5 \
  --query_type rewrite \
  --top_k 10 \
  --output_file results/test3_dense.jsonl

# Test 4: Hybrid
echo "Test 4: Hybrid Retrieval..."
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --dense_model BAAI/bge-base-en-v1.5 \
  --query_type rewrite \
  --top_k 10 \
  --output_file results/test4_hybrid.jsonl

# Test 5: Hybrid + Re-ranking
echo "Test 5: Hybrid + Re-ranking..."
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --dense_model BAAI/bge-base-en-v1.5 \
  --query_type rewrite \
  --use_reranker \
  --top_k 10 \
  --output_file results/test5_hybrid_reranked.jsonl

echo "All tests complete!"
```

### Evaluate All Tests
```bash
cd /Users/pratibharevankar/Desktop/mt-rag-benchmark
source mtrag/bin/activate

for file in experiments/retrieval/results/test*.jsonl; do
  if [[ ! "$file" == *"_evaluated.jsonl" ]]; then
    output="${file%.jsonl}_evaluated.jsonl"
    echo "Evaluating $file..."
    python scripts/evaluation/run_retrieval_eval.py \
      --input_file "$file" \
      --output_file "$output" 2>&1 | tail -3
  fi
done
```

### Compare All Results
```bash
# Create comparison table
echo "Method,R@5,nDCG@5" > experiments/retrieval/results/comparison.csv
for file in experiments/retrieval/results/*_evaluated_aggregate.csv; do
  if [ -f "$file" ]; then
    method=$(basename "$file" | sed 's/_evaluated_aggregate.csv//')
    metrics=$(grep "clapnq\|all" "$file" | tail -1 | awk -F',' '{print $2","$3}')
    echo "$method,$metrics" >> experiments/retrieval/results/comparison.csv
  fi
done

# View comparison
cat experiments/retrieval/results/comparison.csv
```

---

## 🎯 Step-by-Step Testing Workflow

### Phase 1: Verify Basic Functionality

```bash
# 1. Test imports
cd experiments/retrieval
source ../../mtrag/bin/activate
python -c "from base_retriever import BaseRetriever; print('✓ Imports OK')"

# 2. Test corpus loading
python -c "
from corpus_loader import load_corpus, get_corpus_path
import os
os.chdir('../..')
corpus = load_corpus(get_corpus_path('clapnq'))
print(f'✓ Loaded {len(corpus)} documents')
"

# 3. Test query loading
python -c "
from corpus_loader import load_queries, get_queries_path
import os
os.chdir('../..')
queries = load_queries(get_queries_path('clapnq', 'lastturn'))
print(f'✓ Loaded {len(queries)} queries')
"
```

### Phase 2: Test Individual Components

```bash
# Test BM25
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever bm25 \
  --query_type lastturn \
  --top_k 5 \
  --output_file results/test_bm25_small.jsonl

# Test Dense (will download model first time)
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever dense \
  --query_type lastturn \
  --top_k 5 \
  --output_file results/test_dense_small.jsonl

# Test Hybrid
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --query_type lastturn \
  --top_k 5 \
  --output_file results/test_hybrid_small.jsonl
```

### Phase 3: Test Improvements

```bash
# Test with query rewrite
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --query_type rewrite \
  --top_k 10 \
  --output_file results/test_rewrite.jsonl

# Test with enhanced query rewriting
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --query_type rewrite \
  --query_rewrite contextual \
  --top_k 10 \
  --output_file results/test_contextual.jsonl

# Test with re-ranking
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --query_type rewrite \
  --use_reranker \
  --top_k 10 \
  --output_file results/test_reranked.jsonl
```

### Phase 4: Full Evaluation

```bash
cd /Users/pratibharevankar/Desktop/mt-rag-benchmark
source mtrag/bin/activate

# Evaluate all test files
for file in experiments/retrieval/results/test*.jsonl; do
  if [[ ! "$file" == *"_evaluated.jsonl" ]]; then
    output="${file%.jsonl}_evaluated.jsonl"
    python scripts/evaluation/run_retrieval_eval.py \
      --input_file "$file" \
      --output_file "$output"
  fi
done

# View all results
echo "=== All Test Results ==="
for file in experiments/retrieval/results/*_evaluated_aggregate.csv; do
  echo ""
  echo "File: $(basename $file)"
  cat "$file"
done
```

---

## 🔍 Verification Checklist

After running tests, verify:

- [ ] All imports work without errors
- [ ] Corpus loads successfully
- [ ] Queries load successfully
- [ ] BM25 retrieval works
- [ ] Dense retrieval works (may take time to download model)
- [ ] Hybrid retrieval works
- [ ] Re-ranking works (if tested)
- [ ] Evaluation script runs without errors
- [ ] Results show improvement over baseline

---

## 📈 Expected Results

| Test | Expected R@5 | Expected nDCG@5 |
|------|--------------|-----------------|
| BM25 Baseline | 0.19-0.20 | 0.17-0.18 |
| BM25 + Rewrite | 0.24-0.25 | 0.21-0.22 |
| Dense + Rewrite | 0.35-0.37 | 0.32-0.34 |
| Hybrid + Rewrite | 0.40-0.45 | 0.35-0.40 |
| Hybrid + Rewrite + Re-rank | 0.45-0.50 | 0.40-0.45 |

---

## 🐛 Troubleshooting

### Issue: Import Errors
```bash
# Make sure you're in the right directory
cd experiments/retrieval
source ../../mtrag/bin/activate

# Test imports
python -c "from base_retriever import BaseRetriever; print('OK')"
```

### Issue: Model Download Takes Time
```bash
# First run of dense retrieval will download model (~400MB)
# This is normal and only happens once
# Be patient!
```

### Issue: Out of Memory
```bash
# Reduce batch size
--batch_size 16
--reranker_batch_size 16
```

### Issue: Slow Re-ranking
```bash
# Re-ranking is slower but improves quality
# Use GPU if available: --device cuda
# Or skip re-ranking for faster tests
```

### Issue: Results Not Better
```bash
# Make sure you're using:
# 1. query_type rewrite (not lastturn)
# 2. hybrid retriever (not just bm25)
# 3. Check evaluation output carefully
```

---

## 🎓 Quick Test Commands

### Minimal Test (Fastest)
```bash
cd experiments/retrieval && source ../../mtrag/bin/activate
python retrieval_pipeline.py --domain clapnq --retriever bm25 --query_type lastturn --top_k 5 --output_file results/quick_test.jsonl
cd ../.. && python scripts/evaluation/run_retrieval_eval.py --input_file experiments/retrieval/results/quick_test.jsonl --output_file experiments/retrieval/results/quick_test_evaluated.jsonl
cat experiments/retrieval/results/quick_test_evaluated_aggregate.csv
```

### Best Quality Test
```bash
cd experiments/retrieval && source ../../mtrag/bin/activate
python retrieval_pipeline.py --domain clapnq --retriever hybrid --dense_model BAAI/bge-base-en-v1.5 --query_type rewrite --query_rewrite contextual --use_reranker --top_k 10 --output_file results/best_test.jsonl
cd ../.. && python scripts/evaluation/run_retrieval_eval.py --input_file experiments/retrieval/results/best_test.jsonl --output_file experiments/retrieval/results/best_test_evaluated.jsonl
cat experiments/retrieval/results/best_test_evaluated_aggregate.csv
```

---

## 📝 Test Results Template

After testing, document your results:

```markdown
# Test Results - [Date]

## Configuration
- Domain: clapnq
- Retriever: hybrid
- Query Type: rewrite
- Re-ranking: Yes/No

## Results
- R@5: [value]
- nDCG@5: [value]
- Time: [duration]

## Comparison
- Baseline: R@5=0.193
- Improvement: +[X]%
```

---

## ✅ Success Criteria

Your test is successful if:
1. ✅ No errors during execution
2. ✅ Results file is created
3. ✅ Evaluation completes successfully
4. ✅ R@5 > 0.30 (with improvements)
5. ✅ Results show improvement over baseline

Good luck testing! 🎯

