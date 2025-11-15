# Guide: Splitting Passage-Level Data for Retrieval Tasks

## Overview

This guide explains best practices for splitting passage-level data into train/dev/test sets for retrieval tasks in the MTRAG benchmark.

## Key Principles

### 1. **Keep Corpus Intact**
- **DO NOT split the corpus (passages)**
- All passages should remain available for retrieval in all splits
- The corpus in `corpora/passage_level/` is used for all train/dev/test queries
- This simulates real-world retrieval where the entire document collection is available

### 2. **Split Queries and Qrels**
- **Split queries** (`*_lastturn.jsonl`, `*_rewrite.jsonl`, `*_questions.jsonl`)
- **Split qrels** (relevance judgments in `qrels/dev.tsv`)
- These are what you actually train/validate/test on

### 3. **Conversation-Level Splitting**
- **Keep all queries from the same conversation together**
- Prevents data leakage (same conversation shouldn't appear in multiple splits)
- Query IDs contain conversation info: `{author}_{timestamp}<::>{turn_number}`
- Extract conversation ID: `{author}_{timestamp}` (before `<::>`)

### 4. **Stratified Splitting**
- Maintain distribution across:
  - **Domains** (clapnq, cloud, fiqa, govt)
  - **Question types** (if available)
  - **Answerability** (answerable vs unanswerable)
- Ensures each split is representative

## Recommended Split Ratios

### Standard Split (Recommended)
- **Train: 70%** - For training retrieval models
- **Dev: 15%** - For hyperparameter tuning and model selection
- **Test: 15%** - For final evaluation (held out completely)

### Alternative Splits

**Small Dataset (if you have < 500 queries per domain):**
- Train: 60%
- Dev: 20%
- Test: 20%

**Large Dataset (if you have > 2000 queries per domain):**
- Train: 80%
- Dev: 10%
- Test: 10%

## Usage

### Basic Usage

```bash
# Standard 70/15/15 split
python scripts/split_passage_data.py \
    --input_dir human/retrieval_tasks \
    --output_dir human/retrieval_tasks_split \
    --train_ratio 0.7 \
    --dev_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```

### Custom Split Ratios

```bash
# 80/10/10 split
python scripts/split_passage_data.py \
    --input_dir human/retrieval_tasks \
    --output_dir human/retrieval_tasks_split \
    --train_ratio 0.8 \
    --dev_ratio 0.1 \
    --test_ratio 0.1 \
    --seed 42
```

### Split Specific Domains

```bash
# Only split clapnq domain
python scripts/split_passage_data.py \
    --input_dir human/retrieval_tasks \
    --output_dir human/retrieval_tasks_split \
    --domains clapnq \
    --train_ratio 0.7 \
    --dev_ratio 0.15 \
    --test_ratio 0.15
```

### Split Specific Query Types

```bash
# Only split lastturn queries
python scripts/split_passage_data.py \
    --input_dir human/retrieval_tasks \
    --output_dir human/retrieval_tasks_split \
    --query_types lastturn \
    --train_ratio 0.7 \
    --dev_ratio 0.15 \
    --test_ratio 0.15
```

## Output Structure

After splitting, you'll have:

```
human/retrieval_tasks_split/
├── clapnq/
│   ├── clapnq_lastturn_train.jsonl
│   ├── clapnq_lastturn_dev.jsonl
│   ├── clapnq_lastturn_test.jsonl
│   ├── clapnq_rewrite_train.jsonl
│   ├── clapnq_rewrite_dev.jsonl
│   ├── clapnq_rewrite_test.jsonl
│   ├── clapnq_questions_train.jsonl
│   ├── clapnq_questions_dev.jsonl
│   ├── clapnq_questions_test.jsonl
│   └── qrels/
│       ├── train.tsv
│       ├── dev.tsv
│       └── test.tsv
├── cloud/
│   └── ...
├── fiqa/
│   └── ...
└── govt/
    └── ...
```

## Using Split Data

### For Training

```python
from experiments.retrieval.corpus_loader import load_queries, load_corpus

# Load training queries
train_queries = load_queries("human/retrieval_tasks_split/clapnq/clapnq_lastturn_train.jsonl")

# Load full corpus (same for all splits)
corpus = load_corpus("corpora/passage_level/clapnq.jsonl.zip")

# Train your retrieval model...
```

### For Evaluation

```bash
# Evaluate on dev set
python experiments/retrieval/retrieval_pipeline.py \
    --domain clapnq \
    --query_type lastturn \
    --queries_file human/retrieval_tasks_split/clapnq/clapnq_lastturn_dev.jsonl \
    --output_file results/clapnq_dev_results.jsonl

# Evaluate with dev qrels
python scripts/evaluation/run_retrieval_eval.py \
    --input_file results/clapnq_dev_results.jsonl \
    --qrels_file human/retrieval_tasks_split/clapnq/qrels/dev.tsv \
    --output_file results/clapnq_dev_evaluated.jsonl
```

## Important Notes

### 1. **Corpus Remains Unchanged**
- Always use the original corpus files from `corpora/passage_level/`
- Do NOT create separate corpus files for train/dev/test
- This is the correct approach for retrieval tasks

### 2. **Reproducibility**
- Always use a fixed `--seed` for reproducible splits
- Document the seed used in your experiments
- Same seed + same ratios = same splits

### 3. **Data Leakage Prevention**
- The script ensures no conversation appears in multiple splits
- All queries from conversation `{author}_{timestamp}` go to the same split
- This prevents information leakage between splits

### 4. **Evaluation Protocol**
- **Train set**: Use for training (if doing supervised learning)
- **Dev set**: Use for hyperparameter tuning, model selection, early stopping
- **Test set**: Use ONLY for final evaluation, never for tuning

### 5. **Cross-Validation Alternative**
For small datasets, consider k-fold cross-validation:

```python
# Example: 5-fold CV
for fold in range(5):
    # Split data into 5 folds
    # Use 4 folds for train, 1 for validation
    # Rotate which fold is validation
    pass
```

## Statistics

After splitting, check the distribution:

```bash
# Count queries in each split
for split in train dev test; do
    echo "$split:"
    wc -l human/retrieval_tasks_split/clapnq/clapnq_lastturn_${split}.jsonl
done

# Count qrels in each split
for split in train dev test; do
    echo "$split:"
    wc -l human/retrieval_tasks_split/clapnq/qrels/${split}.tsv
done
```

## Troubleshooting

### Issue: "No queries found"
- Check that the input directory structure matches expected format
- Verify domain names are correct (clapnq, cloud, fiqa, govt)
- Check query type names (lastturn, rewrite, questions)

### Issue: "Ratios don't sum to 1.0"
- Ensure train_ratio + dev_ratio + test_ratio = 1.0
- Example: 0.7 + 0.15 + 0.15 = 1.0 ✓

### Issue: Uneven splits
- This is normal for small datasets
- The script splits by conversations, not individual queries
- Some conversations have more queries than others

## Best Practices Summary

1. ✅ **Use conversation-level splitting** to prevent leakage
2. ✅ **Keep corpus intact** - use same corpus for all splits
3. ✅ **Use fixed random seed** for reproducibility
4. ✅ **Standard 70/15/15 split** is a good starting point
5. ✅ **Never tune on test set** - use dev for tuning
6. ✅ **Document your split** - record seed and ratios used
7. ✅ **Check distributions** - ensure splits are balanced

## References

- BEIR Format: https://github.com/beir-cellar/beir
- MTRAG Benchmark: https://github.com/IBM/mt-rag-benchmark
- Information Retrieval Evaluation: https://en.wikipedia.org/wiki/Information_retrieval#Evaluation

