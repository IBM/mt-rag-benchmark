# MTRAG-UN

**[Paper](#paper) | [Corpora](#corpora) | [Retrieval Tasks](#retrieval-tasks) | [Generation Tasks](#generation-tasks)**

We present MTRAG-UN, a benchmark for exploring open challenges in multi-turn retrieval augmented generation. MTRAG-UN consists of 666 tasks containing over 2,800 conversation turns across 6 domains with accompanying corpora. It focuses on four open challenges:

* **UN**answerable Question — the user question cannot be answered from retrieved passages
* **UN**derspecified Question — the user question is ill-formed or ambiguous, lacking the information to determine a clear intent
* **NON**standalone Question — the user question cannot be understood without the prior turns
* **UN**clear Response — the user doesn't understand or disagrees with the model answer and requires clarification

> [!NOTE]
> The Banking and Telco corpora and their associated tasks are coming soon.

## Paper

The paper describing the benchmark and experiments is available on Arxiv:

[MTRAG-UN: A Benchmark for Open Challenges in Multi-Turn RAG Conversations](https://arxiv.org/abs/2602.23184)\
_Sara Rosenthal, Yannis Katsis, Vraj Shah, Lihong He, Lucian Popa, Marina Danilevsky_

## Corpora

MTRAG-UN uses the four corpora from MTRAG (ClapNQ, FiQA, Govt, Cloud) plus two new enterprise corpora:

| Corpus | Domain | # Documents | # Passages |
| ------------- | ------------- | ------------- | ------------- |
| Banking | Banking | 4,497 | 33,380 |
| Telco | Telecommunications | 4,616 | 52,350 |

The four shared corpora are available in the [corpora](../corpora/) folder. The Banking and Telco corpora were created by crawling ~1K web pages per domain to produce sets of inter-connected pages suitable for writing complex conversations.

## Retrieval Tasks

Retrieval tasks are provided in BEIR format. The qrels cover the 468 answerable and partially answerable questions.

| Domain | Qrels |
| ------------- | ------------- |
| ClapNQ | [qrels](retrieval_tasks/qrels/clapnq.tsv) |
| Cloud | [qrels](retrieval_tasks/qrels/cloud.tsv) |
| FiQA | [qrels](retrieval_tasks/qrels/fiqa.tsv) |
| Govt | [qrels](retrieval_tasks/qrels/govt.tsv) |

## Generation Tasks

Tasks are derived from the 666 conversations by selecting one representative turn per conversation (prioritizing challenging UN-turns). For underspecified conversations, the underspecified turn is selected.

| Setting | Description | File |
| ------------- | ------------- | ------------- |
| Reference | Generation using reference passages (up to 10) | [reference.jsonl](generation_tasks/reference.jsonl) |
