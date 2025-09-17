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
