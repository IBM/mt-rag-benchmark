#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# *****************************************************************
# (C) Copyright IBM Corp. 2023 All Rights Reserved.
#
# The source code for this program is not published or otherwise
# divested of its trade secrets, irrespective of what has been
# deposited with the U.S. Copyright Office.
# *****************************************************************

from typing import Dict, List, Tuple
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import os
import sys
import logging
import json
import argparse
import yaml
import pandas as pd
import evaluate
import html
import re
import string
from bs4 import BeautifulSoup

class LABELS:
    ANSWERABLE = "ANSWERABLE"
    UNANSWERABLE = "UNANSWERABLE"

def remove_articles(text: str):
    return re.sub(r"\b(a|an|the)\b", " ", text)


def normalize_white_spaces(text):
    return " ".join([x for x in text.split() if x])


def remove_punc(text):
    exclude = set(string.punctuation)

    return "".join(ch for ch in text if ch not in exclude)

def normalize_text(txt: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    return normalize_white_spaces(remove_articles(remove_punc(txt.lower())))

def clean_html(text: str) -> str:
    return BeautifulSoup(html.unescape(text), features="lxml").get_text()

def strip_newline_words(text: str) -> str:
    text = text.replace("\n", " ")
    return re.sub(r"[^\w\s]", "", text).lower()

# ================================================
#                Third Party scorers
# ================================================
logging.getLogger("absl").setLevel(logging.ERROR)

timestamp = datetime.now()
bertscore_metric = evaluate.load(
    "bertscore", experiment_id=f"bs.{timestamp.strftime('%Y%m%d%H%M%S')}"
)
rouge_evaluator = evaluate.load(
    "rouge", experiment_id=f"rme.{timestamp.strftime('%Y%m%d%H%M%S')}"
)

# ================================================
#                   custom scorers
# ================================================

def recall(prediction: str, target: str, *args, **kwargs) -> float:
    # Tokenize
    prediction_tokens = normalize_text(prediction).split()
    target_tokens = normalize_text(target).split()

    # Identify common tokens
    common_token = Counter(prediction_tokens) & Counter(target_tokens)
    num_common_tokens = sum(common_token.values())

    # Calculate recall
    if num_common_tokens == 0:
        return 0

    return 1.0 * num_common_tokens / len(target_tokens)


def rouge_l(prediction: str, target: str, *args, use_stemmer=False, **kwargs) -> float:
    return rouge_evaluator.compute(
        predictions=[prediction],
        references=[target],
        rouge_types=["rougeL"],
        use_aggregator=False,
        use_stemmer=use_stemmer,
    )["rougeL"][0]


def bertscore(
    prediction: str,
    target: str,
    *args,
    lang: str = "en",
    **kwargs,
) -> Tuple[float, float, float]:

    score = bertscore_metric.compute(
        predictions=[prediction],
        references=[target],
        lang=lang,
        rescale_with_baseline=True,
        model_type="microsoft/deberta-xlarge-mnli",
    )
    return score["f1"][0], score["precision"][0], score["recall"][0]


def bert_prec(prediction: str, target: str, *args, lang: str = "en", **kwargs) -> float:
    _, p, _ = bertscore(prediction, target, lang=lang)
    return p


def bert_recall(
    prediction: str, target: str, *args, lang: str = "en", **kwargs
) -> float:
    _, _, r = bertscore(prediction, target, lang=lang)
    return r


def k_prec(prediction: str, target: str, *args, **kwargs) -> float:
    # Tokenize
    prediction_tokens = normalize_text(prediction).split()
    target_tokens = normalize_text(target).split()

    # Identify common tokens
    common_token = Counter(prediction_tokens) & Counter(target_tokens)
    num_common_tokens = sum(common_token.values())

    # Calculate F1
    if num_common_tokens == 0:
        return 0

    return 1.0 * num_common_tokens / len(prediction_tokens)


def extractiveness_rouge(text: str, target: str, *args, **kwargs) -> float:
    """
    RougeL based extractive measure between text and target

    Args:
        text (str): reference text
        target (str): reference target

    Returns:
        float: extractiveness score
    """
    text = clean_html(strip_newline_words(text))
    target = clean_html(strip_newline_words(target))
    if target.replace(" ", "") in text.replace(" ", ""):
        return 1
    else:
        return rouge_l(text, target)
    
def length(text: str, target: str, *args, **kwargs) -> int:
    return len(text)

def rb_agg(row):
    agg = 0
    
    try:
        recall = (row['metrics']['BertscoreR'][0]+1)/2 
    except:
        recall = row['metrics']['Recall'][0]
    rouge = row['metrics']['RougeL_stemFalse'][0]
    if 'BertKPrec' not in row['metrics']:
        extractiveness = 0
    else:
        extractiveness = (max([0 if value == None else value for value in row['metrics']['BertKPrec']])+1)/2
    agg = 3 * recall * rouge * extractiveness / ((recall * rouge) + (recall * extractiveness) + (rouge * extractiveness))
    row['metrics']['RB_agg'] = [agg]


# Basic logging configurations
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    encoding="utf-8",
    level=logging.INFO,
)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        dest="input",
        help="Path containing file to run",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        dest="output",
        help="Path to save output",
    )
    parser.add_argument(
        "-e",
        "--evaluators",
        type=str,
        dest="evaluators",
        help="Evaluators configuration file",
        default="scripts/evaluation/config.yaml",
    )
    return parser.parse_args()


def read_json_with_pandas(filepath: str) -> pd.DataFrame:
    return pd.read_json(
        filepath,
        lines=filepath.endswith(".jsonl"),
        dtype={"task_id": str, "conversation_id": str},
    )

def score(instance: dict, metrics: Dict[str, dict]) -> List[dict]:

    if "metrics" not in instance:
        instance["metrics"] = {}

    for metric, scorer in metrics.items():
        

        if "target" in scorer["target"]:
            targets = instance["targets"]
        elif scorer["target"] == "passage":
            targets = instance["contexts"]

        if scorer["prediction"] == "prediction":
            prediction = instance["predictions"][0]["text"]
        
        for target in targets:
            if scorer["target"] == "target_label":
                target = target["enrichments"]["answerability"]
            else:
                target = target["text"]

            if metric not in instance["metrics"]:
                instance["metrics"][metric] = []
            if metric == 'RB_agg':
                scorer["func"](instance)
            else:
                instance["metrics"][metric].append(
                    scorer["func"](prediction, target, lang="en")
                )


def process(
    input_file: str,
    metrics_filename: str,
    evaluators: Dict[str, dict],
    logger: logging.Logger
):
    # Step 1: Predictions directory
    
    logger.info("==============================================================")
    logger.info("       Running evaluations for %s", input_file)
    logger.info("==============================================================")
    
    model_predictions = read_json_with_pandas(filepath=f"{input_file}")
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)
    
    with open(metrics_filename, mode="w", encoding="utf-8") as fp:

        # add agg if all necessary evaluators included
        if 'RougeL_stemFalse' in evaluators and 'Recall' in evaluators and 'BertscoreR' in evaluators and 'BertKPrec' in evaluators:
            evaluators["RB_agg"] = {'func': rb_agg, 'prediction': 'prediction', 'target': 'target'}

        # Step 1.c: Run scorers
        progress_bar = tqdm(len(model_predictions))
        for index in range(0, len(model_predictions)):
            instance = model_predictions.iloc[index].to_dict()

            # Score
            score(
                instance=instance,
                metrics=evaluators,
            )

            # Write metrics per task, so if it crashes, you can continue in the middle
            json.dump(instance, fp)
            fp.write("\n")
            fp.flush()
            progress_bar.update(1)

def run_algorithmic_judges(evaluator_file, input_file, output_file):
    logger = logging.getLogger(__name__)
    with open(evaluator_file, "r", encoding="utf-8") as f:
        # Step 1.a: Read
        evaluators = yaml.safe_load(f)

        # Step 1.b: Initializing
        for evaluator in evaluators.values():
            evaluator["func"] = getattr(sys.modules[__name__], evaluator["func"])

        process(input_file=input_file, metrics_filename=output_file, evaluators=evaluators, logger=logger)

    logger.info("Finished evaluating")
    

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    args = parse_args()

    evaluator_file = args.evaluators
    input_file = args.input
    output_file = args.output
    
    run_algorithmic_judges(evaluator_file, input_file, output_file)

