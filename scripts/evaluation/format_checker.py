import json
import argparse
import os
import sys
from collections import Counter

MAX_CONTEXTS = 10

def check_file_size(file_path, max_size_mb=20):
    """
    Check if the file size is less than max_size_mb.
    
    Args:
        file_path (str): Path to the JSONL file.
        max_size_mb (int): Maximum allowed file size in MB.
        
    Returns:
        bool: True if file size is less than max_size_mb, False otherwise.
    """
    
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"File size: {file_size_mb:.2f} MB")
    
    if file_size_mb > max_size_mb:
        print(f"Error: File exceeds {max_size_mb} MB limit.")
        return False
    else:
        print("File size is within the limit.")
        return True
    
    
def validate_json(line: str, line_no: int, errors: list):
    """Check JSON parsing only."""
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        errors.append(f"[Line {line_no}] Invalid JSON: {e}")
        return None

## Retrieval-specific validation functions
def validate_required_fields_retrieval(item: dict, line_no: int, errors: list):
    """Check only the required top-level fields."""
    required = ["task_id", "Collection", "contexts"]

    for field in required:
        if field not in item:
            errors.append(f"[Line {line_no}] Missing required field '{field}'")

    if "task_id" in item and not isinstance(item.get("task_id"), str):
        errors.append(f"[Line {line_no}] 'task_id' must be a string")

    if "Collection" in item and not isinstance(item.get("Collection"), str):
        errors.append(f"[Line {line_no}] 'Collection' must be a string")

    if "contexts" in item and not isinstance(item.get("contexts"), list):
        errors.append(f"[Line {line_no}] 'contexts' must be a list JSON objects, where each item contains 'document_id' and 'text'.")

## Generation-specific validation functions
def validate_required_fields_generation(item: dict, line_no: int, errors: list):
    """Check top-level required fields and types."""
    required = ["task_id", "input", "contexts", "predictions"]

    for field in required:
        if field not in item:
            errors.append(f"[Line {line_no}] Missing required field '{field}'")

    if "task_id" in item and not isinstance(item.get("task_id"), str):
        errors.append(f"[Line {line_no}] 'task_id' must be a string")
    
    if "input" in item and not isinstance(item.get("input"), list):
        errors.append(f"[Line {line_no}] 'input' must be a list of message objects, where each item contains 'speaker' and 'text'.")
    
    if "contexts" in item and not isinstance(item.get("contexts"), list):
        errors.append(f"[Line {line_no}] 'contexts' must be a list JSON objects, where each item contains 'document_id' and 'text'.")
    if "predictions" in item and not isinstance(item.get("predictions"), list):
        errors.append(f"[Line {line_no}] 'predictions' must be a list of prediction objects, each containing 'text'.")


def validate_taskc(item: dict, line_no: int, errors: list):
    """Validate combined retrieval + generation fields without duplication."""
    required = [ "task_id",  "Collection", "input", "contexts", "predictions"]

    for field in required:
        if field not in item:
            errors.append(f"[Line {line_no}] Missing required field '{field}'")

    if "task_id" in item and not isinstance(item.get("task_id"), str):
        errors.append(f"[Line {line_no}] 'task_id' must be a string")

    if "Collection" in item and not isinstance(item.get("Collection"), str):
        errors.append(f"[Line {line_no}] 'Collection' must be a string")
    
    if "input" in item and not isinstance(item.get("input"), list):
        errors.append(f"[Line {line_no}] 'input' must be a list of message objects, where each item contains 'speaker' and 'text'.")
        
    if "contexts" in item and not isinstance(item.get("contexts"), list):
        errors.append(f"[Line {line_no}] 'contexts' must be a list JSON objects, where each item contains 'document_id' and 'text'.")

    if "predictions" in item and not isinstance(item.get("predictions"), list):
        errors.append(f"[Line {line_no}] 'predictions' must be a list of prediction objects, each containing 'text'.")



EMPTY_CONTEXT_LINES = []
def validate_contexts_retrieval(item: dict, line_no: int, errors: list):
    """Validate the contexts list and its required fields."""
    contexts = item.get("contexts")

    if isinstance(contexts, list) and len(contexts) == 0:
        EMPTY_CONTEXT_LINES.append(line_no)
        
    if not isinstance(contexts, list):
        errors.append(f"[Line {line_no}] 'contexts' must be a list")
        return

    if len(contexts) > MAX_CONTEXTS:
        errors.append(f"[Line {line_no}] Too many contexts ({len(contexts)}). Maximum allowed is {MAX_CONTEXTS}.")

    for i, ctx in enumerate(contexts):
        if not isinstance(ctx, dict):
            errors.append(f"[Line {line_no}] contexts[{i}] must be an object")
            continue

        # Check required context fields
        if "document_id" not in ctx:
            errors.append(f"[Line {line_no}] contexts[{i}] missing 'document_id'")
        elif not isinstance(ctx["document_id"], str):
            errors.append(f"[Line {line_no}] contexts[{i}].document_id must be a string")

        if "score" not in ctx:
            errors.append(f"[Line {line_no}] contexts[{i}] missing 'score'")
        elif not isinstance(ctx["score"], (int, float)):
            errors.append(f"[Line {line_no}] contexts[{i}].score must be numeric")
            

def validate_predictions(item: dict, line_no: int, errors: list):
    """Validate the predictions list and expected fields."""
    preds = item.get("predictions", [])
    if not isinstance(preds, list):
        return
    for i, p in enumerate(preds):
        if not isinstance(p, dict):
            errors.append(f"[Line {line_no}] predictions[{i}] must be an object")
            continue
        if "text" not in p:
            errors.append(f"[Line {line_no}] predictions[{i}] missing 'text'")
        elif not isinstance(p["text"], str):
            errors.append(f"[Line {line_no}] predictions[{i}].text must be a string")


def process_line(line: str, line_no: int, errors: list, mode: str):
    line = line.strip()
    if not line:
        return

    item = validate_json(line, line_no, errors)
    if item is None:
        return

    if mode == "retrieval_taska":
        validate_required_fields_retrieval(item, line_no, errors)
        validate_contexts_retrieval(item, line_no, errors)
    
    if mode == "generation_taskb":
        validate_required_fields_generation(item, line_no, errors)
        validate_contexts_retrieval(item, line_no, errors)
        validate_predictions(item, line_no, errors)

    if mode == "rag_taskc":
        validate_taskc(item, line_no, errors)
        validate_contexts_retrieval(item, line_no, errors)
        validate_predictions(item, line_no, errors)


def compare_task_ids(input_file, prediction_file):
    """
    Validate that:
      1. Number of task_ids in prediction file == number in input.
      2. Prediction file contains exactly the same set of task_ids as input.
    Returns a list of errors.
    """
    errors = []

    def read_ids(path):
        ids = []
        with open(path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    # Main validator handles this
                    continue
                tid = item.get("task_id")
                if isinstance(tid, str):
                    ids.append(tid)
        return ids

    input_ids = read_ids(input_file)
    output_ids = read_ids(prediction_file)

    # --- Check 1: number of instances ---
    if len(input_ids) != len(output_ids):
        errors.append(
            f"Mismatch in number of instances: input={len(input_ids)}, output={len(output_ids)}"
        )

    # --- Check 2: same task_id set ---
    input_set = set(input_ids)
    output_set = set(output_ids)

    missing = input_set - output_set
    extra = output_set - input_set

    for tid in missing:
        errors.append(f"Missing task_id '{tid}' in prediction file.")

    for tid in extra:
        errors.append(f"Prediction file contains extra task_id '{tid}' not found in input.")

    return errors


def validate_prediction_file(input_file:str, prediction_file: str, mode: str):
    errors = []

    task_id_errors = compare_task_ids(input_file, prediction_file)
    errors.extend(task_id_errors)

    with open(prediction_file, "r") as f:
        for line_no, line in enumerate(f, start=1):
            process_line(line, line_no, errors, mode)

    warnings = []
    if EMPTY_CONTEXT_LINES:
        warnings.append(
            f"'contexts' is empty on {len(EMPTY_CONTEXT_LINES)} line(s): {EMPTY_CONTEXT_LINES}"
        )

    return errors, warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--prediction_file", required=True)
    parser.add_argument(
        "--mode",
        choices=["retrieval_taska", "generation_taskb", "rag_taskc"],
        required=True,
        help="Specify whether to run the format checker for retrieval eval or generation eval."
    )
    args = parser.parse_args()

    check_file_size(args.prediction_file)
                    
    errors, warnings = validate_prediction_file(args.input_file, args.prediction_file, args.mode)

    print("\n--- Format Check Results ---")
    
    if warnings:
        print(f"Found {len(warnings)} warning(s):")
        for w in warnings:
            print(" -", w)
            
    if not errors:
        print("\nFormat is valid for the eval script.")
    else:
        print(f"Found {len(errors)} issue(s):")
        for e in errors:
            print(" -", e)


if __name__ == "__main__":
    main()
