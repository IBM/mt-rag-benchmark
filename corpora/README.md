Our benchmark is built on document corpora from 4 domains: ClapNQ, Cloud, FiQA and Govt. ClapNQ and FiQA are existing corpora from QA/IR datasets, while Govt and Cloud are new corpora assembled specifically for this benchmark. 

We provide two versions of the corpus, `document_level` and `passage_level`. `document_level` has the original document level offsets and `passage_level` provides the offsets we used during ingestion of the corpus. We *strongly* recommend you use the `passage_level` version of the corpus for all experiments to align with our experiments and reference passages.

> [!IMPORTANT]  
> Download and uncompress the files to use the corpora.

| Corpus | Domain  | Data | # Documents | # Passages |
| ------------- |  ------------- | ------------- | ------------- | ------------- |
|  ClapNQ [[1](https://github.com/primeqa/clapnq)] | Wikipedia | [Corpus](passage_level/clapnq.jsonl.zip) | 4,293 | 183,408  |
|  Cloud | Technical Documentation | [Corpus](passage_level/cloud.jsonl.zip) | 57,638 |  61,022  | 
|  FiQA [[2](https://huggingface.co/datasets/BeIR/fiqa)] | Finance | [Corpus](passage_level/fiqa.jsonl.zip) | 7,661 | 49,607 |
|  Govt | Government  | [Corpus](passage_level/govt.jsonl.zip) | 8,578 | 72,422 |
