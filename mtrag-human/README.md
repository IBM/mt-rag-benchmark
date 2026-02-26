# MTRAG: Human Data

MTRAG consists of 110 multi-turn conversations that are converted to 842 evaluation tasks.

## Features

* diverse question types
* answerable, unanswerable, partial, and conversational questions
* multi-turn: follow-up and clarification
* four domains
* relevant and irrelevant passages (irrelevant passages were not enforced but ones that exist can be used as hard negatives)

## Conversations

We provide our benchmark of 110 conversations in conversation format [HERE](conversations/conversations.json).

They average 7.7 turns per conversation. Each conversation is on a single corpus domain and includes a variety of question types, answerability and multi-turn dimensions. All conversations created by our annotators have gone through a review phase to ensure they are of high quality. During the review phase annotators could accept or reject conversations, and repair responses, passage relevance, and enrichments as needed. They were not allowed to edit the questions or passages as such changes could negatively affect the conversation flow.

## Retrieval Tasks

The retrieval task per domain in BEIR format on the Answerable and Partial tasks only. Additional details are available in the retrieval [README](retrieval_tasks/README.md).

| Name  | Corpus | Queries |
| ------------- |  ------------- | ------------- |
|  ClapNQ |  [Corpus](../corpora/passage_level/clapnq.jsonl.zip) | [Queries](retrieval_tasks/clapnq/) |
|  Cloud |  [Corpus](../corpora/passage_level/cloud.jsonl.zip) | [Queries](retrieval_tasks/cloud/) |
|  FiQA |   [Corpus](../corpora/passage_level/fiqa.jsonl.zip) | [Queries](retrieval_tasks/fiqa/) |
|  Govt |   [Corpus](../corpora/passage_level/govt.jsonl.zip) | [Queries](retrieval_tasks/govt/) |

## Generation Tasks

The conversations are converted into 842 tasks. A task is a conversation turn containing all previous turns together with the last user question (e.g., the task created for turn $k$ includes all user and agent questions/responses for the first $k-1$ turns plus the user question for turn $k$). Our generation tasks measure performance under three retrieval settings. Additional details are available in the generation [README](generation_tasks/README.md).

| Setting  | Description | File |
| ------------- | ------------- |  ------------- |
| Reference  | Generation using reference passages | [reference.jsonl](generation_tasks/reference.jsonl) |
| Reference + RAG | Retrieval followed by generation but with the reference passages kept in the top 5 passages  | [reference+RAG.jsonl](generation_tasks/reference+RAG.jsonl) |
| Full RAG | Retrieval followed by generation where retrieval results consist of the top 5 passages | [RAG.jsonl](generation_tasks/RAG.jsonl) |

### Results

We provide generation results in the [analytics files](evaluations/) for the experiments provided in our paper.

| Setting  | Description | File |
| ------------- | ------------- |  ------------- |
| Reference  | Generation using reference passages | [reference.json](evaluations/reference.json) |
| Reference + RAG | Retrieval followed by generation but with the reference passages kept in the top 5 passages  | [reference+RAG.json](evaluations/reference+RAG.json) |
| Full RAG | Retrieval followed by generation where retrieval results consist of the top 5 passages | [RAG.json](evaluations/RAG.json) |
| Human Evaluation Reference  | Generation using reference passages on a subset with human evaluation | [reference_subset_with_human_evaluations.json](evaluations/reference_subset_with_human_evaluations.json) |
