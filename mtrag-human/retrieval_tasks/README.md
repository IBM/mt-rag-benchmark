
# MT RAG Bench for Retrieval

All retrieval tasks are provided in BEIR format and can be used with BEIR code https://github.com/beir-cellar/beir/ (e.g. see https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/lexical/evaluate_bm25.py)

Each retrieval task requires the corpus, qrels, and questions:

| Domain Name  | Corpus | Qrels | Rewrite Question | Last Question | All Questions |
| ------------- |  ------------- | ------------- | ------------- | ------------- | ------------- | 
|  ClapNQ |  [corpus](/corpora/clapnq.jsonl) | [qrels](clapnq/qrels/dev.tsv) | [queries](clapnq/clapnq_rewrite.jsonl) | [queries](clapnq/clapnq_lastturn.jsonl) | [queries](clapnq/clapnq_questions.jsonl) |
|  Cloud |  [corpus](/corpora/cloud.jsonl) | [qrels](cloud/qrels/dev.tsv) | [queries](cloud/cloud_rewrite.jsonl) | [queries](cloud/cloud_lastturn.jsonl) | [queries](cloud/cloud_questions.jsonl) | [Queries](cloud/qrels/dev.tsv) | 
|  FiQA |   [corpus](/corpora/fiqa.jsonl) | [qrels](fiqa/qrels/dev.tsv) | [queries](fiqa/fiqa_rewrite.jsonl) | [queries](fiqa/fiqa_lastturn.jsonl) | [queries](fiqa/) | [queries](fiqa/fiqa_questions.jsonl) |
|  Govt |   [corpus](/corpora/govt.jsonl) | [qrels](govtqrels/dev.tsv/) | [queries](govt/govt_rewrite.jsonl) | [queries](govt/govt_lastturn.jsonl) | [queries](govt/govt_questions.jsonl) |

>[!NOTE]  
>When processing the input for ingestion into the index we split long contexts into 512 token chunks (with 100 token overlap). In the dev.tsv files you see that chunk id which will have two additional offsets. For example, the first corpus-id in https://github.com/IBM/mt-rag-benchmark/blob/main/human/retrieval_tasks/clapnq/qrels/dev.tsv is  `822086267_7384-8758-0-1374`. The corresponding corpus ID can be found in https://github.com/IBM/mt-rag-benchmark/blob/main/corpora/clapnq.jsonl.zip by searching for `822086267_7384-8758` - dropping the last two values that are the offsets within the correct relevant passage.

In addition to the formats above, we provide a script [scripts/conversations2retrieval.py](/scripts/conversations2retrieval.py) for converting from the conversation format to the beir format. This script can be used to try different input and modified with your own rewrites. An example command for converting to the last turn only is below:

```
python scripts/conversations2retrieval.py 
-i human/conversations/conversations.json
-o human/retrieval_tasks/
-t -1
```

## Results

| Retriever  | Setup | R@1 | R@3 | R@5 | R@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 | 
| ------------- |  ------------- | ------------- | ------------- |  ------------- | ------------- | ------------- |  ------------- | ------------- | ------------- |  
BM25 | Last Turn | 0.08 | 0.15 | 0.20 | 0.27 | 0.17 | 0.16 | 0.18 | 0.21
BM25 | Query Rewrite | 0.09 | 0.18 | 0.25 | 0.33 | 0.20 | 0.19 | 0.22 | 0.25
BGE-base 1.5 |  Last Turn | 0.13 | 0.24 | 0.30 | 0.38 | 0.26 | 0.25 | 0.27 | 0.30
BGE-base 1.5 | Query Rewrite | 0.17 | 0.30 | 0.37 | 0.47 | 0.34 | 0.31 | 0.34 | 0.38
Elser | Last Turn | 0.18 | 0.39 | 0.49 | 0.58 | 0.42 | 0.41 | 0.45 | 0.49
Elser | Query Rewrite | 0.20 | 0.43 | 0.52 | 0.64 | 0.46 | 0.45 | 0.48 | 0.54
