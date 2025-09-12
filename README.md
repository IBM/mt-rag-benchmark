# MTRAG: Multi-Turn RAG Benchmark

**[Paper](#paper) | [Corpora](#corpora) | [Human Data](#human-data) | [Retrieval](#retrieval-tasks) | [Generation](#generation-tasks) | [Synthetic Data](#synthetic-data) | [Getting Started](#getting-started) |  [Contact](#contact)**

We present MTRAG, a comprehensive and diverse human-generated multi-turn RAG dataset, accompanied by four document corpora. To the best of our knowledge, MTRAG is the first end-to-end human-generated multi-turn RAG benchmark that reflects real-world properties of multi-turn conversations.

## Paper

The paper describing the benchmark and experiments is available on Arxiv:

[MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2501.03468)\
_Yannis Katsis, Sara Rosenthal, Kshitij Fadnis, Chulaka Gunasekara, Young-Suk Lee, Lucian Popa, Vraj Shah, Huaiyu Zhu, Danish Contractor, Marina Danilevsky_

If you use MTRAG, please cite the paper as follows:

```
@misc{katsis2025mtrag,
      title={MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems}, 
      author={Yannis Katsis and Sara Rosenthal and Kshitij Fadnis and Chulaka Gunasekara and Young-Suk Lee and Lucian Popa and Vraj Shah and Huaiyu Zhu and Danish Contractor and Marina Danilevsky},
      year={2025},
      eprint={2501.03468},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.03468}, 
}
```

## Corpora

Our benchmark is built on document corpora from 4 domains: ClapNQ, Cloud, FiQA and Govt. ClapNQ and FiQA are existing corpora from QA/IR datasets, while Govt and Cloud are new corpora assembled specifically for this benchmark. 

> [!IMPORTANT]  
> Download and uncompress the files to use the corpora.

| Corpus | Domain  | Data | # Documents | # Passages |
| ------------- |  ------------- | ------------- | ------------- | ------------- |
|  ClapNQ [[1](https://github.com/primeqa/clapnq)] | Wikipedia | [Corpus](corpora/passage_level/clapnq.jsonl.zip) | 4,293 | 183,408  |
|  Cloud | Technical Documentation | [Corpus](corpora/passage_level/cloud.json.zip) | 57,638 |  61,022  | 
|  FiQA [[2](https://huggingface.co/datasets/BeIR/fiqa)] | Finance | [Corpus](corpora/passage_level/fiqa.jsonl.zip) | 7,661 | 49,607 |
|  Govt | Government  | [Corpus](corpora/passage_level/govt.jsonl.zip) | 8,578 | 72,422 |

> [!NOTE]
> Please see the corpora [README](corpora/README.md) regarding using the corpus at passage level (preferred) vs document level.

   
## Human Data

MTRAG consists of 110 multi-turn conversations that are converted to 842 evaluation tasks. 

### ⭐ ⭐ ⭐ Features ⭐ ⭐ ⭐ 

* diverse question types
* answerable, unanswerable, partial, and conversational questions
* multi-turn: follow-up and clarification
* four domains
* relevant and irrelevant passages (irrelevant passages were not enforced but ones that exist can be used as hard negatives)

### Conversations

We provide our benchmark of 110 conversations in conversation format [HERE](human/conversations/conversations.json). 

They average 7.7 turns per conversation. Each conversation is on a single corpus domain and includes a variety of question types, answerability and multi-turn dimensions. All conversations created by our annotators have gone through a review phase to ensure they are of high quality. During the review phase annotators could accept or reject conversations, and repair responses, passage relevance, and enrichments as needed. They were not allowed to edit the questions or passages as such changes could negatively affect the conversation flow.

### Retrieval Tasks

The retrieval task per domain in BEIR format on the Answerable and Partial tasks only. Additional details are available in the retrieval [README](human/retrieval_tasks/README.md).

| Name  | Corpus | Queries |
| ------------- |  ------------- | ------------- |
|  ClapNQ |  [Corpus](corpora/passage_level/clapnq.jsonl.zip) | [Queries](human/retrieval_tasks/clapnq/) |
|  Cloud |  [Corpus](corpora/passage_level/cloud.jsonl.zip) | [Queries](human/retrieval_tasks/cloud/) | 
|  FiQA |   [Corpus](corpora/passage_level/fiqa.jsonl.zip) | [Queries](human/retrieval_tasks/fiqa/) |
|  Govt |   [Corpus](corpora/passage_level/govt.jsonl.zip) | [Queries](human/retrieval_tasks/govt/) |


### Generation Tasks

The conversations are converted into 842 tasks. A task is a conversation turn containing all previous turns together with the last user question (e.g., the task created for turn $k$ includes all user and agent questions/responses for the first $k-1$ turns plus the user question for turn $k$). Our generation tasks measure performance under three retrieval settings.

| Setting  | Description | File |
| ------------- | ------------- |  ------------- |
| Reference  | Generation using reference passages | [reference.jsonl](human/generation_tasks/reference.jsonl) |  
| Reference + RAG | Retrieval followed by generation but with the reference passages kept in the top 5 passages  | [reference+RAG.jsonl](human/generation_tasks/reference+RAG.jsonl) |
| Full RAG | Retrieval followed by generation where retrieval results consist of the top 5 passages | [RAG.jsonl](human/generation_tasks/RAG.jsonl) |

#### Results

We provide generation results in the [analytics files](human/evaluations) for the experiments provided in our paper.

| Setting  | Description | File |
| ------------- | ------------- |  ------------- |
| Reference  | Generation using reference passages | [reference.json](human/evaluations/reference.json) |  
| Reference + RAG | Retrieval followed by generation but with the reference passages kept in the top 5 passages  | [reference+RAG.json](human/evaluations/reference+RAG.json) |
| Full RAG | Retrieval followed by generation where retrieval results consist of the top 5 passages | [RAG.json](human/evaluations/RAG.json) |
| Human Evaluation Reference  | Generation using reference passages on a subset with human evaluation | [reference_subset_with_human_evaluations.json](human/evaluations/reference_subset_with_human_evaluations.json) |  

## Synthetic Data

Manually creating data is an expensive and time consuming process that does not scale well. Automating this process has become popular via synthetic data generation. For further details of creation see this [paper](https://arxiv.org/abs/2409.11500). 

###  Conversations

We provide 200 synthetically generated conversations that follow the properties of the human data. The conversations are available [HERE](synthetic/conversations/)
    
### Generation Tasks

| Setting  | Description | File |
| ------------- | ------------- |  ------------- |
| Reference  | Generation using reference passages. | [synthetic.jsonl](synthetic/generation_tasks/synthetic.jsonl) |  


## Getting Started

### Running Retrieval

Retrieval experiments can be run using the BEIR codebase as described in the retrieval [README](human/retrieval_tasks/README.md). The corpus will need to be ingested to run experiments.

### Running Generation

Generation experiments can be run using any desired models (e.g. available on HuggingFace) and settings as described in the generation [README](human/generation_tasks/README.md). 

### Evaluating Retrieval and Generation

Retrieval and Generation experiments can be evaluated using our evaluation scripts as described in the evaluation [README](scripts/evaluation/README.md).

### Viewing Evaluations

We provide [analytics files](human/evaluations) in InspectorRAGet format, which can be used to inspect the evaluation results and perform further analysis. Load any of the analytics files in [InspectorRAGet](https://huggingface.co/spaces/kpfadnis/InspectorRAGet) by clicking "Visualize" and follow the instructions shown on the screen.

## Acknowledgements

* We'd like to thank our internal annotators for their considerable effort in creating these conversations: Mohamed Nasr, Joekie Gurski, Tamara Henderson, Hee Dong Lee, Roxana Passaro, Chie Ugumori, Marina Variano, Eva-Maria Wolfe 
* We'd like to thank Krishnateja Killamsetty for question classification
* We'd like to thank Lihong He for corpus ingestion
* We'd like to thank Aditya Gaur for deployment help

## Contributors

Sara Rosenthal, Yannis Katsis, Kshitij Fadnis, Chulaka Gunasekara, Young-Suk Lee, Lucian Popa, Vraj Shah, Huaiyu Zhu, Danish Contractor, Marina Danilevsky

## Contact

* Sara Rosenthal sjrosenthal@us.ibm.com
* Yannis Katsis yannis.katsis@ibm.com
* Marina Danilevsky mdanile@us.ibm.com
