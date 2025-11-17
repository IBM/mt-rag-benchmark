# Guide: Splitting Document-Level Corpus

## Overview

This guide explains how to split document-level corpus files into train/dev/test sets using `split_document_corpus.py`.

## When to Use This Script

**Use this script when:**
- Training document encoders
- Fine-tuning models on documents
- Creating separate document collections for different purposes
- Need to split the corpus itself (not queries)

**For retrieval tasks:**
- Typically, you should **keep the corpus intact** and split queries instead
- Use `split_passage_data.py` to split queries and qrels
- The corpus remains available for all splits

## Key Differences from Passage-Level Splitting

| Aspect | Document-Level Splitting | Passage-Level Splitting |
|--------|-------------------------|------------------------|
| **What is split** | Document corpus files | Queries and qrels |
| **Corpus** | Split into train/dev/test | Kept intact |
| **Use case** | Training document models | Retrieval evaluation |
| **Output** | `{domain}_train.jsonl.zip` | `{domain}_{query_type}_train.jsonl` |

## Usage

### Basic Usage

```bash
# Standard 70/15/15 split (compressed)
python scripts/split_document_corpus.py \
    --input_dir corpora/document_level \
    --output_dir corpora/document_level_split \
    --train_ratio 0.7 \
    --dev_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```

### Uncompressed Output

```bash
# Save as uncompressed JSONL files
python scripts/split_document_corpus.py \
    --input_dir corpora/document_level \
    --output_dir corpora/document_level_split \
    --train_ratio 0.7 \
    --dev_ratio 0.15 \
    --test_ratio 0.15 \
    --no_compress
```

### Custom Split Ratios

```bash
# 80/10/10 split
python scripts/split_document_corpus.py \
    --input_dir corpora/document_level \
    --output_dir corpora/document_level_split \
    --train_ratio 0.8 \
    --dev_ratio 0.1 \
    --test_ratio 0.1 \
    --seed 42
```

### Split Specific Domains

```bash
# Only split clapnq domain
python scripts/split_document_corpus.py \
    --input_dir corpora/document_level \
    --output_dir corpora/document_level_split \
    --domains clapnq \
    --train_ratio 0.7 \
    --dev_ratio 0.15 \
    --test_ratio 0.15
```

## Output Structure

After splitting, you'll have:

```
corpora/document_level_split/
├── clapnq_train.jsonl.zip
├── clapnq_dev.jsonl.zip
├── clapnq_test.jsonl.zip
├── cloud_train.jsonl.zip
├── cloud_dev.jsonl.zip
├── cloud_test.jsonl.zip
├── fiqa_train.jsonl.zip
├── fiqa_dev.jsonl.zip
├── fiqa_test.jsonl.zip
├── govt_train.jsonl.zip
├── govt_dev.jsonl.zip
└── govt_test.jsonl.zip
```

## Example Results

Tested on ClapNQ domain:
- **Total documents**: 178,890
- **Train**: 125,222 (70.0%)
- **Dev**: 26,833 (15.0%)
- **Test**: 26,835 (15.0%)

## Using Split Documents

### Loading Split Documents

```python
import zipfile
import json

# Load train documents
with zipfile.ZipFile('corpora/document_level_split/clapnq_train.jsonl.zip', 'r') as z:
    jsonl_file = z.namelist()[0]
    with z.open(jsonl_file) as f:
        for line in f:
            doc = json.loads(line.decode('utf-8'))
            # Process document
            doc_id = doc['_id']
            text = doc['text']
            title = doc.get('title', '')
```

### With Corpus Loader

You can modify the corpus loader to use split files:

```python
from experiments.retrieval.corpus_loader import load_corpus

# Load train corpus
train_corpus = load_corpus('corpora/document_level_split/clapnq_train.jsonl.zip')

# Load dev corpus
dev_corpus = load_corpus('corpora/document_level_split/clapnq_dev.jsonl.zip')
```

## Important Notes

### 1. **Random Splitting**
- Documents are randomly shuffled before splitting
- Use a fixed `--seed` for reproducibility
- Same seed + same ratios = same splits

### 2. **Document Independence**
- Unlike query splitting, documents are split independently
- No conversation-level grouping (documents don't have conversations)
- Each document is assigned to one split only

### 3. **File Sizes**
- Document-level files are large (178K+ documents for ClapNQ)
- Compression is recommended (default: enabled)
- Uncompressed files can be very large

### 4. **Retrieval Tasks**
- For retrieval evaluation, **don't split the corpus**
- Keep corpus intact and split queries instead
- Use `split_passage_data.py` for retrieval tasks

## Comparison: Document vs Passage Splitting

### Document-Level Splitting (`split_document_corpus.py`)
- **Input**: `corpora/document_level/*.jsonl.zip`
- **Output**: `corpora/document_level_split/{domain}_{split}.jsonl.zip`
- **Purpose**: Training document models
- **Splits**: Documents themselves

### Passage-Level Splitting (`split_passage_data.py`)
- **Input**: `human/retrieval_tasks/*/`
- **Output**: `human/retrieval_tasks_split/{domain}/{domain}_{query_type}_{split}.jsonl`
- **Purpose**: Retrieval evaluation
- **Splits**: Queries and qrels (corpus stays intact)

## Troubleshooting

### Issue: "Corpus file not found"
- Check that input directory contains `{domain}.jsonl.zip` files
- Verify domain names are correct (clapnq, cloud, fiqa, govt)

### Issue: "Ratios don't sum to 1.0"
- Ensure train_ratio + dev_ratio + test_ratio = 1.0
- Example: 0.7 + 0.15 + 0.15 = 1.0 ✓

### Issue: Large file sizes
- Use `--compress` (default) to reduce file sizes
- Consider splitting only specific domains if needed

## Best Practices

1. ✅ **Use fixed random seed** for reproducibility
2. ✅ **Compress output files** to save space
3. ✅ **Document your split** - record seed and ratios used
4. ✅ **For retrieval tasks** - use passage-level splitting instead
5. ✅ **Check file sizes** - ensure you have enough disk space

## References

- Passage-Level Splitting: `scripts/SPLITTING_GUIDE.md`
- Corpus README: `corpora/README.md`
- MTRAG Benchmark: https://github.com/IBM/mt-rag-benchmark

