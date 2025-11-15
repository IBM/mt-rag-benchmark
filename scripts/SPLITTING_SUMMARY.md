# Passage-Level Data Splitting - Summary

## Best Method for Splitting Passage-Level Data

### ✅ Recommended Approach: Conversation-Level Stratified Split

**Key Principles:**
1. **Keep corpus intact** - All passages remain available for all splits
2. **Split queries and qrels** - Only split the evaluation data
3. **Conversation-level splitting** - Keep all queries from same conversation together
4. **Stratified by domain** - Maintain distribution across domains

### Standard Split Ratios

- **Train: 70%** - For training retrieval models
- **Dev: 15%** - For hyperparameter tuning and model selection  
- **Test: 15%** - For final evaluation (held out completely)

### Why This Method?

1. **Prevents Data Leakage**
   - All queries from conversation `{author}_{timestamp}` go to same split
   - No information from test conversations leaks into training

2. **Realistic Evaluation**
   - Corpus stays intact (simulates real-world retrieval)
   - All passages available for retrieval in all splits

3. **Reproducible**
   - Fixed random seed ensures same splits every time
   - Easy to share and compare results

4. **Maintains Distribution**
   - Stratified splitting keeps domain/question type balance
   - Each split is representative of the full dataset

## Quick Start

```bash
# Activate virtual environment
source mtrag/bin/activate

# Split all domains and query types (70/15/15)
python scripts/split_passage_data.py \
    --input_dir human/retrieval_tasks \
    --output_dir human/retrieval_tasks_split \
    --train_ratio 0.7 \
    --dev_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```

## Test Results

Tested on ClapNQ domain with `lastturn` queries:

```
Total: 208 queries from 29 conversations
Split: 20 train, 4 dev, 5 test conversations
Queries: 143 train (68.8%), 31 dev (14.9%), 34 test (16.3%)
```

Note: Slight variation from 70/15/15 is expected because we split by conversations, not individual queries.

## Output Structure

```
human/retrieval_tasks_split/
├── clapnq/
│   ├── clapnq_lastturn_train.jsonl
│   ├── clapnq_lastturn_dev.jsonl
│   ├── clapnq_lastturn_test.jsonl
│   └── qrels/
│       ├── train.tsv
│       ├── dev.tsv
│       └── test.tsv
└── [other domains...]
```

## Usage After Splitting

### Training
```python
# Load training queries
train_queries = load_queries("human/retrieval_tasks_split/clapnq/clapnq_lastturn_train.jsonl")

# Load full corpus (same for all splits)
corpus = load_corpus("corpora/passage_level/clapnq.jsonl.zip")
```

### Evaluation
```bash
# Evaluate on dev set
python experiments/retrieval/retrieval_pipeline.py \
    --queries_file human/retrieval_tasks_split/clapnq/clapnq_lastturn_dev.jsonl \
    --output_file results/clapnq_dev.jsonl

# Evaluate with dev qrels
python scripts/evaluation/run_retrieval_eval.py \
    --input_file results/clapnq_dev.jsonl \
    --qrels_file human/retrieval_tasks_split/clapnq/qrels/dev.tsv \
    --output_file results/clapnq_dev_evaluated.jsonl
```

## Important Notes

1. **Corpus files remain unchanged** - Always use original corpus from `corpora/passage_level/`
2. **Never tune on test set** - Use dev for tuning, test only for final evaluation
3. **Use fixed seed** - Document the seed used for reproducibility
4. **Check distributions** - Verify splits are balanced across domains

## Files Created

- `scripts/split_passage_data.py` - Main splitting script
- `scripts/SPLITTING_GUIDE.md` - Comprehensive guide with examples
- `scripts/SPLITTING_SUMMARY.md` - This summary document

## References

- See `scripts/SPLITTING_GUIDE.md` for detailed documentation
- BEIR Format: https://github.com/beir-cellar/beir
- MTRAG Benchmark: https://github.com/IBM/mt-rag-benchmark

