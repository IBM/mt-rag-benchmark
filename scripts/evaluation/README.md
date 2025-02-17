
This is a standalone script to run the evaluation metrics reported in the paper. It expects as input a file in our generation format (e.g. `human/generation_tasks/reference.jsonl`) with a single prediction output for each task in the following format:

```
 "predictions": [
    {
      "text": "ANSWER TEXT HERE",
      "class": "ANSWERABLE|UNANSWERABLE"
    }
  ]
```


### Requirements

Create a new conda environment and pip install the following packages

```
tqdm
beautifulsoup4
lxml
pandas
evaluate==0.4.1
bert_score
rouge-score
```

## Run

The `scripts/evaluation/responses-10.jsonl` is sample input with predictions on the first 10 reference tasks.

```
python /dccstor/srosent1/human_ai_eval/mt-rag-benchmark/scripts/evaluation/run_scoring.py -i scripts/evaluation/responses-10.jsonl -o <output_path> -e scripts/evaluation/config.yaml
```