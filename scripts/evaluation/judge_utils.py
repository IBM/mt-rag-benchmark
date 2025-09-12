import pandas as pd
import ast
import os
from tqdm import tqdm
import re

def read_json_with_pandas(filepath: str) -> pd.DataFrame:
    return pd.read_json(
        filepath,
        lines=filepath.endswith(".jsonl"),
        dtype={"task_id": str, "conversation_id": str},
    )


def remove_keys_from_df(df: pd.DataFrame, keys_to_remove: list) -> pd.DataFrame:
    return df.drop(columns=keys_to_remove, errors="ignore")


def extract_conversation(conversation_list):
    if isinstance(conversation_list, str):
        conversation_list = ast.literal_eval(conversation_list)
    
    return "\n".join([f"{turn['speaker']}: {turn['text']}" for turn in conversation_list])

def extract_document_texts(doc_list):
    if isinstance(doc_list, str): 
        doc_list = ast.literal_eval(doc_list)
    
    return [doc['text'] for doc in doc_list]

def extract_texts(text_list):
    if isinstance(text_list, str):
        text_list = ast.literal_eval(text_list)
    
    # return [item['text'] for item in text_list]
    return text_list[0]['text']


def extract_reference(conversation_list):
    if isinstance(conversation_list, str):
        conversation_list = ast.literal_eval(conversation_list)
    
    return conversation_list[0]['text']


def update_or_create_dict(col1, col2_value, key_val):
    if not isinstance(col1, dict) or col1 is None: 
        col1 = {}
    col1[key_val] = [col2_value]
    return col1


def extract_target(val_lst):
    if isinstance(val_lst, str): 
        val_lst = ast.literal_eval(val_lst)
    
    # print(len(val_lst))
    # print(val_lst[0]['text'])
    return val_lst[0]['text']

def split_conversation(inquiry):
    lines = inquiry.splitlines()
    
    user_questions = [line for line in lines if line.startswith("user:")]
    
    if len(user_questions) == 1:
        return "", user_questions[0].split("user: ")[1].strip()
    else:
        last_user_question = user_questions[-1].split("user: ")[1].strip()
        previous_conversation = "\n".join(lines[:-1])
        return previous_conversation, last_user_question
    
    
def format_conversation_radbench(df):
    formatted_conversations = []
    
    for index, row in df.iterrows():
        document_lst = row['document']
        document_txt = "\n\n".join(document_lst)
        
        formatted_conversation = f"""
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question given the provided document and a reference answer.

Your evaluation should assess the faithfulness, appropriateness, and completeness. Your evaluation should focus on the assistant's answer to the question of the current turn.  You will be given the assistant's answer and a reference answer. You will also be given the user questions and assistant's answers of the previous turns of the conversation. You should consider how well the assistant's answer captures the key information, knowledge points mentioned in the reference answer, and how it respects or builds upon the focus and knowledge points from the previous turns. 

[Faithfulness]: You are given the full conversation, the question of the current turn, the assistant's answer, and documents. You should evaluate how faithful is the assistant's answer to the information in the document and previous conversation.
[Appropriateness]: You should evaluate if the assistant's answer is relevant to the question of the current turn and if it addresses all the issues raised by the question without adding extra information.
[Completeness]: You should evaluate whether the assistant's answer is complete with information from the documents.

Begin your evaluation by comparing the assistant's answer against the reference answer in this turn. Be as objective as possible, and provide a detailed justification for your rating. After providing your explanation, you must rate the response on a scale of 1 to 10, strictly following this format: "Rating: [[rating]]", for example: "Rating: [[5]]".

[The Start of Previous Conversation]
{row['previous_conversation']}
[The End of Previous Conversation]

[The Start of Current Turn Question]
{row['current_question']}
[The End of Current Turn Question]

[The Start of Reference Answer]
{row['reference_answer']}
[The End of Reference Answer]

[The Start of Assistant's Answer]
{row['response']}
[The End of Assistant's Answer]

[The Start of Document]
{document_txt}
[The End of Document]
"""
        formatted_conversations.append(formatted_conversation)

    return formatted_conversations



def extract_rating(text):
    
    text = re.sub(r'for example:.*?$', '', str(text), flags=re.IGNORECASE | re.MULTILINE)
    if text != text:
        return 0.5
    
    regex_patterns = [
        r"Rating: \[\[(\d+)\]\]",                       # "Rating: [[5]]"
        r"Rating: (\d+)",                               # "Rating: 10"
        r"\[\[(\d+)\]\]",                               # "[[10]]"
        r"\[(\d+)\]",                                   # "[7]"
        r"Rating: \[\[([0-9]*\.?[0-9]+)\]\]",           # "Rating: [[8.5]]"
        r"Overall Rating: ([0-9]*\.?[0-9]+)",           # "Overall Rating: ([0-9]*\.?[0-9]+)"
        r"Score: ([0-9]*\.?[0-9]+)/10",                 # "Score: 8.6/10"
        r"Score: ([0-9]*\.?[0-9]+)",                    # "Score: 9"
        r"\[Rating\]\n(\d+)",                            # "[Rating]\n7"
        r"Rating: \[\[(\d+)/10\]\]",                     # "Rating: [[7/10]]"
        r"Rating: \[\[([-+]?\d*\.?\d+)",                 # "Rating: [[8"
        r"([-+]?\d*\.?\d+) out of 10"                    # "6 out of 10"

    ]
    
    ret_val = 0.5
    for pattern in regex_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            ret_val = float(matches[0]) / 10
    
    if ret_val <= 0 or ret_val > 1:
        ret_val = 0.5

    return ret_val



def first_token_idk(text: str) -> str:
    if not isinstance(text, str):
        return None
    match = re.match(r"\s*([A-Za-z]+)", text)
    if not match: return "unknown"
    token = match.group(1).lower()
    # return token if token in {"yes", "no", "partial"} else "unknown"
    
    if token == "yes": return 1
    elif token == "no": return 0
    else: return 0.5


def format_idk_judge(df):
    formatted_conversations = []
    
    for index, row in df.iterrows():
        inquiry, response = row['inquiry'], row['response']
        
        prompt_template = f"""Read an inquiry and a response to it.  Decide if the response contains, in part or in whole, an indication that there is not sufficient information to fully answer the inquiry.  Use only one of the following labels: yes, no, partial.  Answer "yes" if the response indicates that the inquiry cannot be answered, for example, due to lack of information (in a reference document or text).  Answer "partial" if the response indicates insufficient information (in a reference document or text) to answer part of the inquiry, but still attempts to answer other part of the inquiry that can be answered.  Answer "no" if the response does not indicate a lack of information preventing answering part of the inquiry, even if some reasoning steps or general knowledge are needed to address the inquiry.

[Inquiry]
how do i enroll my wife in medical insurance?

[Response]
She can continue medical and dental coverage under COBRA by calling the Benefits Center.

[Output]
no


[Inquiry]
What is the Off-Contract Spend app used for?

[Response]
I do not have information regarding the Off-Contract Spend app.

[Output]
yes


[Inquiry]
I need a service for my daughter but they are not offering a discount anymore. Do you know why?

[Response]
I do not see a reference to that in the information, but here is a site where you my find out more about it

[Output]
partial


[Inquiry]
Should I focus first on Word Error Rate or Sentence Error Rate?,

[Response]
The document does not provide a specific answer to your question. However, it mentions that running the experiment against the STT Base Model with no adaptation will give you both Word Error Rate (WER) and Sentence Error Rate (SER). It suggests that analyzing these metrics will help identify areas for improvement. Based on this information, it may be beneficial to focus on both WER and SER to gain a comprehensive understanding of the performance and areas that need improvement.

[Output]
partial


[Inquiry]
What entails Central Purchasing?

[Response]
I don't know.

[Output]
yes


[Inquiry]
Where can I find the list of Bupa Health Assessment Centers

[Response]
I am sorry, but the provided document does not contain answer to your question

[Output]
yes


[Inquiry]
How should I ensure my device is compliant with beekeeper?

[Response]
I'm sorry, but I don't have any information about beekeeper or how to ensure device compliance with it. The provided document does not mention anything about beekeeper. Is there anything else I can help you with?

[Output]
yes


[Inquiry]
How can I learn SQL without taking a course?

[Response]
I’m unable to provide an answer regarding how you can learn SQL without taking a course. However, I can assist with the No SQL databases topic.

[Output]
yes


[Inquiry]
Do these reimbursements get taxed?

[Response]
Your reimbursements from the HCSA won't be taxed.

[Output]
no


[Inquiry]
How do I know if a unit of work is preemptable?

[Response]
I am unable to answer your question specifically, but I can give you some information about preemptable versus non-preemptable units of work. A non-preemptable unit of work can be interrupted, but must receive control after the interrupt is processed, while a preemptable unit of work returns control to the operating system once the interrupt handling completes. 

[Output]
partial


[Inquiry]
What does SKU stand for?

[Response]
I'm sorry, I can't tell you what SKU stands for.

[Output]
yes


[Inquiry]
What is natural language processing (NLP) and how can I build a sentiment analysis model using Python and the NLTK library?

[Response]
Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and human (natural) languages. More specifically, it involves the development of algorithms and statistical models that can understand, interpret, generate, and make sense of human language in a valuable way. However, the document does not describe how to build a sentiment analysis model using Python and the NLTK library.

[Output]
partial


[Inquiry]
How does photosynthesis take place in plants?

[Response]
The provided document does not describe how photosynthesis take place in plants.  However, from my general knoweldge, photosynthesis takes place in plants with following steps: 1. Light absorption: Plants absorb light energy from the sun using chlorophyll, a pigment found in organelles called chloroplasts.  2. Water splitting: The absorbed light energy is used to power a process called photolysis, which splits water molecules into oxygen gas and hydrogen ions.  3. Carbon fixation: The hydrogen ions are used to convert carbon dioxide into a usable form, while the oxygen gas is released as a byproduct.  4. Reduction: The carbon dioxide is reduced to a molecule containing carbon and a high-energy electron, which is then used to produce glucose and other sugars.

[Output]
no


[Inquiry]
User: Which president served for the shortest amount of time?
Agent: I don't have sufficient information to answer that particular question. It depends on the definition of "shortest" and the time frame considered. Some presidents served for only a few months, while others served for many years. I recommend looking it up online or in a reliable reference source.
User:  Which U.S. president served the longest term?

[Response]
Franklin D. Roosevelt, who served as the 32nd President of the United States from 1933 to 1945. 

[Output]
no


[Inquiry]
Which phone has a larger screen?

[Response]
The document does not say which of the phones have a larger sceen.  However, the document states that A has a screen size of 6.3 inches, and B has a screen size of 5.9 inches.  From this we can conclude that A has a larger screen than B.

[Output]
no


[Inquiry]
{inquiry}

[Response]
{response}

[Output]
"""
        formatted_conversations.append(prompt_template)
    return formatted_conversations