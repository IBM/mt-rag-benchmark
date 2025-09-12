from typing import Type, List, Dict, Union, Tuple
import os
import pytrec_eval
import argparse
import csv, json
from judge_utils import *

def evaluate(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k_values: List[int],
                 ignore_identical_ids: bool=True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        
        if ignore_identical_ids:
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0
        
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        # evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_string, recall_string})
        scores = evaluator.evaluate(results)
        
        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)

        return scores, ndcg, _map, recall, precision
    

def compute_results(results, qrels):

    k_values = [1, 3, 5]
    if len(results) == 0:
        ndcg = _map = recall = precision = mrr = {i: '-' for i in k_values}
    else:
        scores_per_query_id, ndcg, _map, recall, precision = evaluate(qrels, results, k_values)

    scores_global = {}
    scores_global[f"nDCG"] = list(ndcg.values())
    scores_global[f"Recall"] = list(recall.values())
    
    return scores_global, scores_per_query_id
   
def load_qrels(qrels_file):
    
    reader = csv.reader(open(qrels_file, encoding="utf-8"), 
                        delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)
    
    qrels = {}
    for id, row in enumerate(reader):
        query_id, corpus_id, score = row[0], row[1], int(row[2])

        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score
 
    return qrels
 
def prepare_results_dict(input_file):
    results = {}
    collection_results = {}
    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            query_id = item["task_id"]
            
            doc_scores = {}
            for ctx in item.get("contexts", []):
                doc_id = ctx["document_id"]
                score = ctx["score"]
                doc_scores[doc_id] = score
            
            results[query_id] = doc_scores
            collection_results[query_id] = item["Collection"]
            
    return results, collection_results


def enrich_json_retrieval(input_file, scores_per_instance, output_file):
 
    retrieval_predictions_pd = read_json_with_pandas(filepath=f"{input_file}")
    
    retrieval_predictions_pd['retriever_scores'] = retrieval_predictions_pd['task_id'].map(scores_per_instance)
    retrieval_predictions_pd["retriever_scores"] = retrieval_predictions_pd["retriever_scores"].apply(lambda x: {} if pd.isna(x) else x)

    retrieval_predictions_pd.to_json(output_file, orient="records", lines=True)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSON file")
    
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    
    retrieval_predictions, collection_results = prepare_results_dict(input_file)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    scores_global_lst = []
    global_scores_per_query_id = {}
    collections = set(collection_results.values())
    
    for collection_name in collections:
        print("\ncollection_name:", collection_name)

        if collection_name == "mt-rag-clapnq-elser-512-100-20240503":
            qrels_file = os.path.join(script_dir, "../../human/retrieval_tasks_convid/clapnq/qrels/dev.tsv")
        if collection_name == "mt-rag-govt-elser-512-100-20240611":
            qrels_file = os.path.join(script_dir, "../../human/retrieval_tasks_convid/govt/qrels/dev.tsv")
        if collection_name == "mt-rag-fiqa-beir-elser-512-100-20240501":
            qrels_file = os.path.join(script_dir, "../../human/retrieval_tasks_convid/fiqa/qrels/dev.tsv")
        if collection_name == "mt-rag-ibmcloud-elser-512-100-20240502":
            qrels_file = os.path.join(script_dir, "../../human/retrieval_tasks_convid/cloud/qrels/dev.tsv")
            
        qrels = load_qrels(qrels_file)
        
        preds_for_collection = {
            qid: retrieval_predictions[qid]
            for qid, coll in collection_results.items()
            if coll == collection_name
        }
        

        scores_global, scores_per_query_id = compute_results(preds_for_collection, qrels)
        scores_global['collection'] = collection_name
        scores_global['count'] = len(preds_for_collection)
        
        print("Retriever Evaluation Aggregate Scores:", scores_global)
        
        global_scores_per_query_id.update(scores_per_query_id)
        scores_global_lst.append(scores_global)

    
    n = len(scores_global_lst[0]['Recall'])
    total_count = sum(d['count'] for d in scores_global_lst)

    weighted_avg_recall, weighted_avg_ndcg = [], []
    for i in range(n):
        weighted_sum_recall = sum(d['Recall'][i] * d['count'] for d in scores_global_lst)
        weighted_avg_recall.append(weighted_sum_recall / total_count)
        
        weighted_sum_ndcg = sum(d['nDCG'][i] * d['count'] for d in scores_global_lst)
        weighted_avg_ndcg.append(weighted_sum_ndcg / total_count)

    print("Weighted average Recall:", weighted_avg_recall)  
    print("Weighted average nDCG:", weighted_avg_ndcg)  

    rows = scores_global_lst.copy()
    rows.append({
        "nDCG": weighted_avg_ndcg,
        "Recall": weighted_avg_recall,
        "collection": "all",
        "count": total_count
    })

    df = pd.DataFrame(rows)
    df.to_csv(f"{os.path.splitext(output_file)[0]}_aggregate.csv", index=False)
    
    enrich_json_retrieval(input_file, global_scores_per_query_id, output_file)

if __name__ == "__main__":
    
    main()
    
    