# MTRAG SemEval 2026 Team Repository

Fork of IBM's Multi-Turn Retrieval-Augmented Generation (MTRAG) benchmark.  
This repository contains our team's experiments for the **SemEval 2026 Multi-Turn RAG** competition.

## Structure
- `experiments/` – our retrieval and generation models
  - `retrieval/` – lexical, dense, and hybrid retrievers
  - `generation/` – RAG-based generation models
- `corpora/`, `human/`, `synthetic/` – original MTRAG datasets
- `scripts/` – evaluation and helper scripts from IBM

## Setup
```bash
git clone https://github.com/pratirvce/MTRAG-Eval.git
cd MTRAG-Eval
pip install -r experiments/retrieval/requirements.txt
```

## Task A: Retrieval Only

See `experiments/retrieval/README.md` for detailed instructions on running the retrieval system.

### Quick Start
```bash
cd experiments/retrieval
python retrieval_pipeline.py --retriever hybrid --query_type rewrite --collection clapnq --top_k 100
```

## About

Evaluating Multi-Turn RAG Conversations
