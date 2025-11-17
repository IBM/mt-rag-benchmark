# MTRAG SemEval 2026 Team Repository

Fork of IBM’s Multi-Turn Retrieval-Augmented Generation (MTRAG) benchmark.  
This repository contains our team's experiments for the **SemEval 2026 Multi-Turn RAG** competition.

## Structure
- `experiments/` – our retrieval and generation models
  - `retrieval/` – lexical, dense, and hybrid retrievers
  - `generation/` – RAG-based generation models
- `corpora/`, `human/`, `synthetic/` – original MTRAG datasets
- `scripts/` – evaluation and helper scripts from IBM

## Setup
```bash
git clone https://github.com/Umit-Azirakhmet/mt-rag-benchmark.git
cd mt-rag-benchmark
pip install -r requirements.txt
