import pandas as pd
import numpy as np
import ast
import os
from datasets import Dataset 
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from judge_utils import *

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai.embeddings import AzureOpenAIEmbeddings

from huggingface_client import HuggingFaceLLMClient
from azure_openai_client import AzureOpenAIClient

from datasets import Dataset
from typing import List, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from langchain.callbacks.manager import CallbackManagerForLLMRun
from ragas.metrics import faithfulness, context_recall, context_precision, answer_relevancy
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from langchain.llms.base import LLM
import warnings
warnings.filterwarnings('ignore')

import torch
import gc

def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# ================================================
# Local LLM class for running RAGAS locally
# ================================================
class LocalLLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, mode_name_or_path: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path,attn_implementation="flash_attention_2", device_map="auto",  torch_dtype="bfloat16", 
                                                            load_in_4bit=True,
                                                            bnb_4bit_compute_dtype=torch.bfloat16,
                                                          )
        self.model.generation_config = GenerationConfig.from_pretrained(mode_name_or_path)

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(model_inputs.input_ids, 
                                            max_new_tokens=2048, 
                                            attention_mask= model_inputs["attention_mask"], pad_token_id=self.tokenizer.pad_token_id)
        
        # generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        # response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        input_length = model_inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_length:]
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        
        return response

    @property
    def _llm_type(self):
        return "chat"

# ================================================
# Get IDK conditioning score
# ================================================
def get_idk_score(row, use_metric):
    answerability_vals = row.get("answerability", [])
    metrics = row.get("metrics", {})

    answerability = answerability_vals[0] if answerability_vals else None
    idk_eval = metrics.get("idk_eval")[0]
    rl_f = metrics.get(use_metric)[0]

    # print(f"Answerability: {answerability}, IDK Eval: {idk_eval}, metric: {rl_f}")

    if answerability in ["UNANSWERABLE", "CONVERSATIONAL"] and idk_eval == 1:
        return 1
    elif answerability in ["UNANSWERABLE", "CONVERSATIONAL"] and idk_eval in [0, 0.5]:
        return 0
    elif idk_eval == 1:
        return 0
    else:
        return rl_f
    
    
def get_idk_conditioned_metrics(input_file, output_file):
    model_predictions = read_json_with_pandas(filepath=f"{input_file}")

    model_predictions['RL_F_idk'] = model_predictions.apply(get_idk_score, axis=1, use_metric = 'RL_F')
    model_predictions['RB_llm_idk'] = model_predictions.apply(get_idk_score, axis=1, use_metric = 'RB_llm')
    model_predictions['RB_agg_idk'] = model_predictions.apply(get_idk_score, axis=1, use_metric = 'RB_agg')
    
    model_predictions['metrics'] = model_predictions.apply(lambda row: update_or_create_dict(row.get('metrics'), row['RL_F_idk'], 'RL_F_idk'), axis=1)
    model_predictions['metrics'] = model_predictions.apply(lambda row: update_or_create_dict(row.get('metrics'), row['RB_llm_idk'], 'RB_llm_idk'), axis=1)
    model_predictions['metrics'] = model_predictions.apply(lambda row: update_or_create_dict(row.get('metrics'), row['RB_agg_idk'], 'RB_agg_idk'), axis=1)

    keys_to_remove = ["RL_F_idk", "RB_llm_idk", "RB_agg_idk"]
    model_predictions = remove_keys_from_df(model_predictions, keys_to_remove)

    model_predictions.to_json(output_file, orient="records", lines=True)
    

# ================================================
# Compute RAGAS Locally
# ================================================
def run_ragas_judges_local(judge_model, input_file, output_file):
    clear_cuda()
    model_predictions = read_json_with_pandas(filepath=f"{input_file}")
    
    model_predictions['inquiry'] = model_predictions['input'].apply(extract_conversation)
    model_predictions['document'] = model_predictions['contexts'].apply(extract_document_texts)
    model_predictions['response'] = model_predictions['predictions'].apply(extract_texts)
    
    data_samples = {}

    data_samples['question'] = model_predictions['inquiry'].values.tolist()
    data_samples['answer'] = model_predictions['response'].values.tolist()
    data_samples['contexts'] = model_predictions['document'].values.tolist()
    dataset = Dataset.from_dict(data_samples)
    
    run_config = RunConfig(timeout=10000, max_workers= 1)
    
    model = LangchainLLMWrapper(LocalLLM(judge_model), run_config)

    score = evaluate(
        dataset,
        metrics=[faithfulness],
        llm=model,
        run_config=run_config,
    )

    df_score = score.to_pandas()

    model_predictions['RL_F'] = df_score['faithfulness'].values
    
    if 'metrics' not in model_predictions:
        model_predictions['metrics'] = None

    model_predictions['metrics'] = model_predictions.apply(lambda row: update_or_create_dict(row.get('metrics'), row['RL_F'], 'RL_F'), axis=1)

    keys_to_remove = ["inquiry", "document", "response", "RL_F"]
    model_predictions = remove_keys_from_df(model_predictions, keys_to_remove)

    model_predictions.to_json(output_file, orient="records", lines=True)
    
    

# ================================================
# Compute RAGAS w/ OpenAI
# ================================================
def run_ragas_judges_openai(input_file, output_file, openai_key, azure_host):

    llm = AzureChatOpenAI(
        deployment_name="gpt-4o-mini-2024-07-18",
        openai_api_base=azure_host,
        openai_api_version="2024-09-01-preview",
        openai_api_key=openai_key, 
        timeout=120 
    )

    # azure_embeddings = AzureOpenAIEmbeddings(
    #     openai_api_version="2024-08-01-preview",
    #     azure_endpoint=azure_host, 
    #     model= "text-embedding-ada-002-2", 
    # )
    
    model_predictions = read_json_with_pandas(filepath=f"{input_file}")

    model_predictions['inquiry'] = model_predictions['input'].apply(extract_conversation)
    model_predictions['document'] = model_predictions['contexts'].apply(extract_document_texts)
    model_predictions['response'] = model_predictions['predictions'].apply(extract_texts)

    data_samples = {}

    data_samples['question'] = model_predictions['inquiry'].values.tolist()
    data_samples['answer'] = model_predictions['response'].values.tolist()
    data_samples['contexts'] = model_predictions['document'].values.tolist()

    dataset = Dataset.from_dict(data_samples)

    run_config = RunConfig(timeout=120) 

    score = evaluate(
        dataset,
        llm=llm,
        # embeddings=azure_embeddings,
        metrics=[
            faithfulness,
            ],
        run_config = RunConfig(timeout=120)
        )
    df_score = score.to_pandas()

    model_predictions['RL_F'] = df_score['faithfulness'].values

    if 'metrics' not in model_predictions:
        model_predictions['metrics'] = None

    model_predictions['metrics'] = model_predictions.apply(lambda row: update_or_create_dict(row.get('metrics'), row['RL_F'], 'RL_F'), axis=1)

    keys_to_remove = ["inquiry", "document", "response", "RL_F"]
    model_predictions = remove_keys_from_df(model_predictions, keys_to_remove)

    model_predictions.to_json(output_file, orient="records", lines=True)
    
# ================================================
# Run Radbench Judge
# ================================================
def run_radbench_judge(judge_model, input_file, output_file):
    model_predictions = read_json_with_pandas(filepath=f"{input_file}")

    model_predictions['inquiry'] = model_predictions['input'].apply(extract_conversation)
    model_predictions['document'] = model_predictions['contexts'].apply(extract_document_texts)
    model_predictions['response'] = model_predictions['predictions'].apply(extract_texts)

    model_predictions['reference_answer'] = model_predictions['targets'].apply(extract_reference)
    model_predictions['previous_conversation'], model_predictions['current_question'] = zip(*model_predictions['inquiry'].apply(split_conversation))    
    
    user_inputs = format_conversation_radbench(model_predictions)
    
    if judge_model == "openai":
        model_name_lst = ['gpt-4o-mini-2024-07-18']
    else:
        model_name_lst = [judge_model]
    
    for model_name in model_name_lst:
        
        if model_name.startswith("gpt-"):
            client = AzureOpenAIClient('gpt-4o-mini-2024-07-18')
        else:
            clear_cuda()
            client = HuggingFaceLLMClient(model_name)
        
        output_lst = ['' for i in range(len(user_inputs))]
        
        i=0
        for user_input in tqdm(user_inputs):
            output = client.generate_response(user_input)
            output_lst[i] = output
            i += 1
    
        model_predictions[f'{model_name}_raw'] = output_lst
        model_predictions[f'{model_name}'] = model_predictions[f'{model_name}_raw'].apply(extract_rating)

    model_predictions['RB_llm'] = model_predictions[model_name_lst].apply(np.median, axis=1)
    
    for model_name in model_name_lst:
        model_predictions = remove_keys_from_df(model_predictions, [f'{model_name}_raw', f'{model_name}'])
    
    if 'metrics' not in model_predictions:
        model_predictions['metrics'] = None

    model_predictions['metrics'] = model_predictions.apply(lambda row: update_or_create_dict(row.get('metrics'), row['RB_llm'], 'RB_llm'), axis=1)
    
    keys_to_remove = ["inquiry", "document", "response", "reference_answer", "previous_conversation", "current_question", "RB_llm"]
    model_predictions = remove_keys_from_df(model_predictions, keys_to_remove)
    
    model_predictions.to_json(output_file, orient="records", lines=True)

# ================================================
# Run IDK Judge
# ================================================
def run_idk_judge(model_name, input_file, output_file):    
    
    if model_name == "openai":
        client = AzureOpenAIClient('gpt-4o-mini-2024-07-18')
    else:
        clear_cuda()
        client = HuggingFaceLLMClient(model_name)
        
    model_predictions = read_json_with_pandas(filepath=f"{input_file}")
    
    model_predictions['inquiry'] = model_predictions['input'].apply(extract_conversation)
    model_predictions['response'] = model_predictions['predictions'].apply(extract_texts)

    formatted_conversations = format_idk_judge(model_predictions)
    
    response_lst = []
    for cur_prompt in tqdm(formatted_conversations):
        if model_name == "openai":
            response = client.generate_response(cur_prompt)
        else:
            response = client.generate_response(cur_prompt, max_new_tokens = 3)
        response_lst.append(response)
            
    model_predictions['idk_eval'] = response_lst
    model_predictions["idk_eval"] = model_predictions["idk_eval"].apply(first_token_idk)

    if 'metrics' not in model_predictions:
            model_predictions['metrics'] = None

    model_predictions['metrics'] = model_predictions.apply(lambda row: update_or_create_dict(row.get('metrics'), row['idk_eval'], 'idk_eval'), axis=1)
    
    model_predictions = remove_keys_from_df(model_predictions, ["inquiry", "response", "idk_eval"])
    model_predictions.to_json(output_file, orient="records", lines=True)
