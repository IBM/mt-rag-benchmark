## Evaluation 

This page describes how to run evaluation for retrieval and generation on MTRAG.

### Requirements

To run the retrieval and generation evaluation, create a new conda environment and install necessary dependencies from `requirements.txt` using

```
pip install -c scripts/evaluation/constraints.txt -r scripts/evaluation/requirements.txt
```

> [!TIP]
> If your installation fails due to torch missing because of `flash_attn`, run a `pip install torch==2.8.0` and then re-run the above requirements.

All evaluation scripts run on top of our file format as provided in `human/generation_tasks/*.jsonl`

### Format Checker Script

We provide a standalone script to validate whether your prediction JSONL file follows the required evaluation formats for all three tasks. The checker supports three modes depending on the evaluation task:
* `retrieval_taska` - validates the fields required by the retrieval evaluation
* `generation_taskb` - validates the fields required by the generation evaluation
* `rag_taskc` - validates the fields required by the RAG evaluation

```
python scripts/evaluation/format_checker.py --input_file <INPUT_FILE> --prediction_file <PREDICTION_FILE> --mode <retrieval_taska|generation_taskb|rag_taskc>
```

Sample input files are available at `scripts/evaluation/sample_data`. These sample files correspond to the first 10 examples from the MT-RAG dataset and can be used to verify that your method produces correctly formatted outputs before running the full evaluation.

You can easily construct valid input files for the format checker and evaluation scripts by modifying any of the files under `human/generation_tasks/*.jsonl`. This an be achieved by removing fields such as `enrichments`, `targets`, and `predictions`. The prediction file does not include fields such as `targets` and `enrichments`.

Sample Input data format for Task A, C:
```
{
  "conversation_id": "dd6b6ffd177f2b311abe676261279d2f",
  "task_id": "dd6b6ffd177f2b311abe676261279d2f::2",
  "Collection": "mt-rag-clapnq-elser-512-100-20240503",
  "input": [
    {
      "speaker": "user",
      "text": "where do the arizona cardinals play this week"
    }
  ]
}
```

Sample prediction file for Task A and input data format for Task B:
```
{
  "conversation_id": "dd6b6ffd177f2b311abe676261279d2f",
  "task_id": "dd6b6ffd177f2b311abe676261279d2f::2",
  "Collection": "mt-rag-clapnq-elser-512-100-20240503",
  "input": [
    {
      "speaker": "user",
      "text": "where do the arizona cardinals play this week"
    }
  ]
  "contexts":
    [
        {
            "document_id": "822086267_7384-8758-0-1374",
            "text": "...",
            "score": 27.759
        }, ...
    ],
}
```

Sample prediction file for Task B and Task C:
```
{
  "conversation_id": "dd6b6ffd177f2b311abe676261279d2f",
  "task_id": "dd6b6ffd177f2b311abe676261279d2f::2",
  "Collection": "mt-rag-clapnq-elser-512-100-20240503",
  "input": [
    {
      "speaker": "user",
      "text": "where do the arizona cardinals play this week"
    }
  ]
  "contexts":
    [
        {
            "document_id": "822086267_7384-8758-0-1374",
            "text": "...",
            "score": 27.759
        }, ...
    ],
    "predictions":
    [
        {
            "text": "..."
        }
    ]
}
```

Sample output format:

```
File size: ...
[File size is within the limit.] or [Error: File exceeds 20 MB limit.]

--- Format Check Results ---
[Found 1 warning(s) ...]
[Format is valid for the eval script.] or [Found x issues: ...]
```

### Retrieval Evaluation

The retrieval script looks at the `contexts` field, which is a list of JSON objects per task. `document_id` and `score` are required for retrieval evaluation and the rest are optional. All fields are necessary for the generation script.

```
"contexts":
    [
        {
            "document_id": "822086267_6698-7277-0-579",
            "source": "",
            "score": 18.759138,
            "text": "2017 Arizona Cardinals season\nOn December 13 , 2016 , the NFL announced that the Cardinals will play the Los Angeles Rams as one of the NFL International Series at Twickenham Stadium in London , England ...",
            "title": "2017 Arizona Cardinals season"
        }, ...
    ],
```

#### Evaluation Script

This script evaluates retrieval performance using Recall and nDCG metrics on a per-collection basis. It also aggregates results across collections and computes weighted averages.

```
python scripts/evaluation/run_retrieval_eval.py --input_file <INPUT_FILE> --output_file <OUTPUT_FILE>
```

Arguments
* input_file: Path to a JSONL file containing retrieval predictions (e.g., `human/generation_tasks/RAG.jsonl`) 
  Each JSON object must contain: 
    - `contexts`: List of retrieval predictions each with format {'document_id': ... , 'score': ...}
    - `Collection` Name of the collection (one of):
        * mt-rag-clapnq-elser-512-100-20240503 
        * mt-rag-govt-elser-512-100-20240611
        * mt-rag-fiqa-beir-elser-512-100-20240501
        * mt-rag-ibmcloud-elser-512-100-20240502

* output_file: Path where evaluation results will be saved. The script appends results under a new `retriever_scores` attribute. The evaluation script also produces a CSV file (`<OUTPUT_FILE_NAME>_aggregate.csv`) with the aggregate results under the same output file directory. 


### Generation Evaluation

This is a standalone script to run the evaluation metrics reported in the paper. It expects as input a file in our generation format (e.g. `human/generation_tasks/reference.jsonl`) which for each task also includes an additional new `predictions` attribute representing the generated LLM response for that task using the following format:

```
 "predictions": [
    {
      "text": "ANSWER TEXT HERE",
    }
  ]
```


The `scripts/evaluation/responses-10.jsonl` is sample input with predictions on the first 10 reference tasks.

To run OpenAI GPT4o-mini as Judge

```
python scripts/evaluation/run_generation_eval.py -i <INPUT_FILE> -o <OUTPUT_FILE> -e scripts/evaluation/config.yaml --provider openai --openai_key <OPENAI_KEY> --azure_host <AZURE_ENDPOINT>
```

> [!TIP]
> Our implementation for evaluting with GPT assumes an Azure endpoint. If you are using an alternate endpoint, you will need to modify the client [see here](azure_openai_client.py#L8). 


To run HuggingFace model as Judge

```
python scripts/evaluation/run_generation_eval.py -i <INPUT_FILE> -o <OUTPUT_FILE> -e scripts/evaluation/config.yaml --provider hf --judge_model ibm-granite/granite-3.3-8b-instruct
```


Arguments
* input_file: Path to a JSONL file containing predictions from the generative model under `predictions`.
* output_file: Path to the output file, which would contain all the evaluated metrics under `metrics`
* OpenAI key and Azure Endpoint if provider is openai
* Huggingface model name if provider is hf

Please see [paper](https://arxiv.org/abs/2501.03468) for the explanation of the metrics.
