# Task A: Retrieval System

This directory contains the implementation for **Task A: Retrieval Only** of the SemEval 2026 Multi-Turn RAG competition.

## Overview

The retrieval system supports three types of retrievers:
- **BM25**: Lexical retrieval using BM25 algorithm
- **Dense**: Semantic retrieval using sentence transformers
- **Hybrid**: Combination of BM25 and dense retrieval

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for BM25)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Quick Start

### 1. Prepare Corpora

The corpora are already available in `corpora/passage_level/` as zip files:
- `clapnq.jsonl.zip`
- `cloud.jsonl.zip`
- `fiqa.jsonl.zip`
- `govt.jsonl.zip`

**Note:** The system automatically handles zip files - no need to unzip!

### 2. Run Retrieval

#### BM25 Retrieval
```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever bm25 \
  --query_type lastturn \
  --top_k 10 \
  --output_file results/clapnq_bm25_lastturn.jsonl
```

#### Dense Retrieval
```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever dense \
  --dense_model BAAI/bge-base-en-v1.5 \
  --query_type lastturn \
  --top_k 10 \
  --output_file results/clapnq_dense_lastturn.jsonl
```

#### Hybrid Retrieval (Recommended)
```bash
python retrieval_pipeline.py \
  --domain clapnq \
  --retriever hybrid \
  --dense_model BAAI/bge-base-en-v1.5 \
  --fusion_method rrf \
  --query_type lastturn \
  --top_k 10 \
  --output_file results/clapnq_hybrid_lastturn.jsonl
```

### 3. Evaluate Results

```bash
python ../../scripts/evaluation/run_retrieval_eval.py \
  --input_file results/clapnq_hybrid_lastturn.jsonl \
  --output_file results/clapnq_hybrid_lastturn_evaluated.jsonl
```

## Usage Examples

### Run on All Domains

```bash
# BM25 on all domains
for domain in clapnq cloud fiqa govt; do
  python retrieval_pipeline.py \
    --domain $domain \
    --retriever bm25 \
    --query_type lastturn \
    --output_file results/${domain}_bm25.jsonl
done
```

### Try Different Query Types

```bash
# Last turn only
python retrieval_pipeline.py --domain clapnq --retriever hybrid \
  --query_type lastturn --output_file results/clapnq_lastturn.jsonl

# All questions
python retrieval_pipeline.py --domain clapnq --retriever hybrid \
  --query_type questions --output_file results/clapnq_questions.jsonl

# With query rewriting
python retrieval_pipeline.py --domain clapnq --retriever hybrid \
  --query_type lastturn --query_rewrite expand \
  --output_file results/clapnq_rewritten.jsonl
```

### Try Different Models

```bash
# BGE models
python retrieval_pipeline.py --domain clapnq --retriever dense \
  --dense_model BAAI/bge-base-en-v1.5 \
  --output_file results/clapnq_bge_base.jsonl

python retrieval_pipeline.py --domain clapnq --retriever dense \
  --dense_model BAAI/bge-large-en-v1.5 \
  --output_file results/clapnq_bge_large.jsonl

# E5 models
python retrieval_pipeline.py --domain clapnq --retriever dense \
  --dense_model intfloat/e5-base-v2 \
  --output_file results/clapnq_e5_base.jsonl
```

## Command Line Arguments

### Required Arguments
- `--domain`: Domain name (clapnq, cloud, fiqa, govt)
- `--retriever`: Retriever type (bm25, dense, hybrid)
- `--output_file`: Output file path for results

### Optional Arguments

**Data:**
- `--corpus_path`: Path to corpus file (default: auto-detect)
- `--queries_path`: Path to queries file (default: auto-detect)
- `--query_type`: Query type (lastturn, questions, rewrite)

**Retriever Configuration:**
- `--dense_model`: Dense model name (default: BAAI/bge-base-en-v1.5)
- `--bm25_k1`: BM25 parameter k1 (default: 1.5)
- `--bm25_b`: BM25 parameter b (default: 0.75)
- `--alpha`: Weight for dense in hybrid (default: 0.5)
- `--fusion_method`: Fusion method for hybrid (rrf, weighted)

**Query Processing:**
- `--query_rewrite`: Query rewriting method (last_turn, all_turns, expand, full)

**Retrieval:**
- `--top_k`: Number of documents to retrieve (default: 10)

**Other:**
- `--device`: Device for dense retrieval (cuda/cpu, default: auto)
- `--batch_size`: Batch size for dense retrieval (default: 32)

## File Structure

```
retrieval/
├── base_retriever.py      # Base retriever interface
├── bm25_retriever.py      # BM25 lexical retriever
├── dense_retriever.py     # Dense semantic retriever
├── hybrid_retriever.py   # Hybrid retriever (BM25 + Dense)
├── corpus_loader.py       # Corpus and query loading utilities
├── query_processor.py     # Query processing and rewriting
├── retrieval_pipeline.py # Main pipeline script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Collection Names

The system automatically maps domains to collection names:
- `clapnq` → `mt-rag-clapnq-elser-512-100-20240503`
- `cloud` → `mt-rag-ibmcloud-elser-512-100-20240502`
- `fiqa` → `mt-rag-fiqa-beir-elser-512-100-20240501`
- `govt` → `mt-rag-govt-elser-512-100-20240611`

## Output Format

Results are saved in JSONL format compatible with the evaluation script:

```json
{
  "task_id": "query_id",
  "Collection": "mt-rag-clapnq-elser-512-100-20240503",
  "contexts": [
    {
      "document_id": "doc_id",
      "score": 0.95,
      "text": "Document text...",
      "title": "Document title"
    }
  ]
}
```

## How to Improve Scores

**Yes, you can significantly improve retrieval scores!** See [IMPROVEMENTS.md](IMPROVEMENTS.md) for:
- Quick wins that can improve R@5 from 0.193 to 0.37 (+91%)
- Advanced techniques for even better performance
- Expected improvements for each strategy
- Step-by-step improvement guide

**Quick improvement:** Use `--query_type rewrite` and `--retriever hybrid` to get ~0.40 R@5!

## Complete Command Reference

For a comprehensive list of all commands, see [COMMANDS.md](COMMANDS.md).

## Tips for Better Performance

1. **Use Hybrid Retrieval**: Combines strengths of lexical and semantic retrieval
2. **Try Query Rewriting**: Experiment with different query processing methods
3. **Tune BM25 Parameters**: Adjust k1 and b for your corpus
4. **Use Better Dense Models**: Try larger models like `BAAI/bge-large-en-v1.5`
5. **Experiment with Query Types**: Try `questions` instead of `lastturn` for multi-turn context
6. **Re-ranking**: Consider adding a re-ranker after initial retrieval

## Baseline Performance

Expected baseline performance (from benchmark):
- BM25 (Last Turn): R@5=0.20, nDCG@5=0.18
- BGE-base 1.5 (Last Turn): R@5=0.30, nDCG@5=0.27
- Elser (Last Turn): R@5=0.49, nDCG@5=0.45

Try to beat these baselines!

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` for dense retrieval
- Use smaller dense models
- Process domains separately

### Slow Retrieval
- Use GPU for dense retrieval (`--device cuda`)
- Consider using FAISS for faster similarity search (optional)
- Reduce `--top_k` during development

### Missing NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Next Steps

1. Run retrieval on all domains
2. Evaluate results using evaluation script
3. Experiment with different configurations
4. Try advanced techniques (re-ranking, query expansion, etc.)
5. Prepare for test phase submission

