# MTRAG: Multi-Turn RAG Benchmark

**[Papers](#papers) | [Corpora](#corpora) | [MTRAG - Human Data](#mtrag---human-data) | [MTRAG - Synthetic Data](#mtrag---synthetic-data) | [MTRAG-UN](#mtrag-un) | [MTRAGEval](#mtrageval) | [Getting Started](#getting-started) | [Contact](#contact)**

We present MTRAG, a comprehensive and diverse human-generated multi-turn RAG dataset, accompanied by four document corpora. To the best of our knowledge, MTRAG is the first end-to-end human-generated multi-turn RAG benchmark that reflects real-world properties of multi-turn conversations.

## Papers

The papers describing the benchmarks and experiments are available on Arxiv:

[MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems](https://doi.org/10.1162/TACL.a.19)\
_Yannis Katsis, Sara Rosenthal, Kshitij Fadnis, Chulaka Gunasekara, Young-Suk Lee, Lucian Popa, Vraj Shah, Huaiyu Zhu, Danish Contractor, Marina Danilevsky_\
Transactions of the Association for Computational Linguistics, 2025

[MTRAG-UN: A Benchmark for Open Challenges in Multi-Turn RAG Conversations](https://arxiv.org/abs/2602.23184)\
_Sara Rosenthal, Yannis Katsis, Vraj Shah, Lihong He, Lucian Popa, Marina Danilevsky_

If you use MTRAG, please cite the paper as follows:

```
@article{katsis2025mtrag,
      title={MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems},
      author={Yannis Katsis and Sara Rosenthal and Kshitij Fadnis and Chulaka Gunasekara and Young-Suk Lee and Lucian Popa and Vraj Shah and Huaiyu Zhu and Danish Contractor and Marina Danilevsky},
      journal={Transactions of the Association for Computational Linguistics},
      volume={13},
      pages={784--808},
      year={2025},
      doi={10.1162/TACL.a.19},
      url={https://doi.org/10.1162/TACL.a.19},
}
```

If you use MTRAG-UN, please cite the paper as follows:

```
@misc{rosenthal2026mtragun,
      title={MTRAG-UN: A Benchmark for Open Challenges in Multi-Turn RAG Conversations},
      author={Sara Rosenthal and Yannis Katsis and Vraj Shah and Lihong He and Lucian Popa and Marina Danilevsky},
      year={2026},
      eprint={2602.23184},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.23184},
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

## MTRAG - Human Data

MTRAG consists of 110 multi-turn conversations that are converted to 842 evaluation tasks, spanning conversations, retrieval tasks, and generation tasks across four domains.

See the [mtrag-human README](mtrag-human/README.md) for full details.

## MTRAG - Synthetic Data

We provide 200 synthetically generated conversations and generation tasks that follow the properties of the human data.

See the [mtrag-synthetic README](mtrag-synthetic/README.md) for full details.

## MTRAG-UN

MTRAG-UN is a benchmark for exploring open challenges in multi-turn RAG, focusing on UNanswerable, UNderspecified, and NONstandalone questions and UNclear responses. It consists of 666 tasks across 6 domains, including two new enterprise corpora (Banking and Telco).

See the [mtragun-human README](mtragun-human/README.md) for full details.

## MTRAGEval

[MTRAGEval](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/) is a task for Evaluating Multi-Turn RAG Conversations at [SemEval 2026](https://semeval.github.io/SemEval2026/). MTRAG is the training data and MTRAG-UN is the evaluation benchmark.

Sample data from MTRAG in the format used by the [evaluation scripts](scripts/evaluation/) is available at [`scripts/evaluation/sample_data`](scripts/evaluation/sample_data/).

## Getting Started

### Running Retrieval

Retrieval experiments can be run using the BEIR codebase as described in the retrieval [README](mtrag-human/retrieval_tasks/README.md). The corpus will need to be ingested to run experiments.

### Running Generation

Generation experiments can be run using any desired models (e.g. available on HuggingFace) and settings as described in the generation [README](mtrag-human/generation_tasks/README.md).

### Evaluating Retrieval and Generation

Retrieval and Generation experiments can be evaluated using our evaluation scripts as described in the evaluation [README](scripts/evaluation/README.md).

### Viewing Evaluations

We provide [analytics files](mtrag-human/evaluations) in InspectorRAGet format, which can be used to inspect the evaluation results and perform further analysis. Load any of the analytics files in [InspectorRAGet](https://huggingface.co/spaces/kpfadnis/InspectorRAGet) by clicking "Visualize" and follow the instructions shown on the screen.

## Acknowledgements

* We'd like to thank our internal annotators for their considerable effort in creating these conversations: Mohamed Nasr, Joekie Gurski, Tamara Henderson, Hee Dong Lee, Roxana Passaro, Chie Ugumori, Marina Variano, Eva-Maria Wolfe
* We'd like to thank Krishnateja Killamsetty for question classification
* We'd like to thank Lihong He for corpus ingestion
* We'd like to thank Aditya Gaur for deployment help

## Contributors

Sara Rosenthal, Yannis Katsis, Kshitij Fadnis, Chulaka Gunasekara, Young-Suk Lee, Lucian Popa, Vraj Shah, Huaiyu Zhu, Lihong He, Danish Contractor, Marina Danilevsky

## Contact

* Sara Rosenthal sjrosenthal@us.ibm.com
* Yannis Katsis yannis.katsis@ibm.com
* Marina Danilevsky mdanile@us.ibm.com
