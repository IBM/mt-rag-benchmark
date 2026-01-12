# MTRAGEval

🎉 Welcome to MTRAGEval! MTRAGEval is a task for Evaluating Multi-Turn RAG Conversations at [SemEval 2026](https://semeval.github.io/SemEval2026/). 🎉

## 📝 Registration

Please fill out this form to register for our task if you plan to participate in MTRAGEval: [Registration Form](https://forms.gle/XgjqgDGgG7wgCb5r6)

## Join Our Mailing List!

[MTRAGEval Mailing List](https://groups.google.com/g/mtrageval)

## Training and Trial Data

The MTRAG Benchmark is released as the trial, training, and validation data for MTRAGEval. You can access the full dataset [here](https://github.com/IBM/mt-rag-benchmark/). We will release new evaluation data during the evalution phase.

Note: The MTRAG Benchmark includes metadata that describes dimensions for each task including the question type (e.g. factoid), answerability (e.g. unanswerable, answerable), and multi-turn type (e.g. follow-up, clarification). This information will *NOT* be provided during evaluation. We will only provide the corpus domain e.g. ClapNQ, Govt.

## 📋 Tasks

### Task A: Retrieval Only
  
*Input*: You are given a set of tasks, where each task contains (a) a conversation comprising of a set of user question/agent response turns ending with a user question and (b) the corresponding document corpus.

*Output*: For each task, you are asked to return an ordered list of 10 passages from the document corpus that are relevant to the last user question (with more relevant passages appearing earlier in the list). Note that your submission for this task will only be evaluated on the subset of answerable questions; however, to avoid information leak for the other tasks, you will not be provided beforehand with info on which questions are answerable or not. 

Note: For task A, we will produce results @ 1, 3, 5, 10 so you should make sure to return 10 contexts.

### Task B: Generation with Reference Passages (Reference)
*Input*: You are given a set of tasks, where each task contains (a) a conversation comprising of a set of user question/agent response turns ending with a user question and (b) a set of relevant passages for the last user question.

*Output*: For each task, you are asked to generate an agent response for the last user question (which should be faithful w.r.t. the relevant passages).

### Task C: Generation with Retrieved Passages (RAG)
 
*Input*: You are given a set of tasks, where each task contains (a) a conversation comprising of a set of user question/agent response turns ending with a user question and (b) the corresponding document corpus.
 
*Output*: For each task, you are asked to first retrieve up to 10 passages from the documents corpus that are relevant to the user question and use them to generate an agent response for the last user question (which should be faithful w.r.t. the retrieved passages). 

*Note*: Your submission for Task C will be evaluated mainly based on the generated agent response; the intermediate list of retrieved passages is part of the evaluation for faithfulness. We allow a maximum of 10 passages to be returned but you do not need to use the full amount (in our experiments in the MTRAG paper we used 5). Returning more contexts may reduce the faithfulness score. We recommend returning at most 5 as used in our paper.

Read more about our tasks in our [proposal](../MT_RAG_SemEval_Proposal.pdf)

## Data Format

The evaluation data follows the MTRAG data format. Input and Output format for the task, sample data, and Format Checker scripts are available on the GitHub repo! Please visit the evaluation [README](https://github.com/IBM/mt-rag-benchmark/blob/main/scripts/evaluation/README.md) for more information.

## Evaluation 

### Evaluation Scripts

Evaluation and Format Checker scripts are available on the GitHub repo! Please visit the evaluation [README](https://github.com/IBM/mt-rag-benchmark/blob/main/scripts/evaluation/README.md) for more information.

### 🏆 Leaderboard Ranking

We will use the Evaluation Scripts provided above to evaluate each team's system. 

The ranking for the retrieval Task A will be using nDCG.

The ranking for the generation Tasks: B and C will be computed as the harmonic mean of RL_F, RB_llm and RB_alg. We present the ranking of the results provided in the MTRAG paper to illustrate the ranking. Please note, that ranking is not the only indication of a strong system. In particular, the difference in rank may be large if several systems achieve close scores as in the results provided below.

|Rank | Task B (Reference) | Harmonic Mean |Rank	|		Task C (RAG) | Harmonic Mean	|	
| -- | -- | -- | -- | -- | -- |
|	1st	|	Reference	|	0.89	|	1st	|	Reference	|	0.81	|
|	2nd	|	GPT-4o	|	0.60	|	2nd	|	GPT-4o	|	0.53	|
|	2nd	|	Llama-3.1-405B-Instruct	|	0.60	|	2nd	|	Llama-3.1-405B-Instruct	|	0.53	|
|	4th	|	GPT-4o-mini	|	0.57	|	4th	|	Qwen-2.5-(72B)	|	0.52	|
|	4th	|	Qwen-2.5-(72B)	|	0.57	|	4th	|	Llama-3.1-70B-Instruct	|	0.52	|
|	4th	|	Command-R+(104B)	|	0.57	|	6th	|	GPT-4o-mini	|	0.51	|
|	7th	|	Qwen-2.5-(7B)	|	0.55	|	6th	|	Command-R+(104B)	|	0.51	|
|	8th	|	Llama-3.1-70B-Instruct	|	0.54	|	6th	|	Qwen-2.5-(7B)	|	0.51	|
|	9th	|	Mixtral-8x22B-Instruct	|	0.51	|	9th	|	Mixtral-8x22B-Instruct	|	0.48	|
|	10th	|	Llama-3.1-8B-Instruct	|	0.45	|	10th	|	Llama-3.1-8B-Instruct	|	0.45	|

## Task Submission and Evaluation Data

Task Submission will only be open during the evaluation phase. All submissions will be via a google form that will be provided per task.

Because LLM judge evaluation is resource-intensive, we're restricting submission to one run per task per team (if you submit more than one we will only evaluate the last one). We will release the predictions after the evaluation phase so that you can try out and report other techniques in your paper.

The evaluation data will be provided to all registered participants at the start of each evaluation phase.

Note: The MTRAG Benchmark includes metadata that describes dimensions for each task including the question type (e.g. factoid), answerability (e.g. unanswerable, answerable), and multi-turn type (e.g. follow-up, clarification). This information will NOT be provided during evaluation. We will only provide the corpus domain e.g. ClapNQ, Govt.

## 📆 Timeline (Tentative)

* Sample and Training data ready 15 July 2025
* Evaluation start 12 January 2026 Task A and C
* Evaluation end 20 January 2026 Task A and C
* Evaluation start 26 January 2026 Task B
* Evaluation end by 2 February 2026 Task B
* Paper submission due February 2026 (Tentative)
* Notification to authors March 2026 (Tentative)
* Camera ready due April 2026 (Tentative)
* SemEval workshop Summer 2026 (co-located with [ACL 2026](https://2026.aclweb.org/))


<!-- ## Task A: Retrieval Only

Given the conversations, use the following [script](/scripts/conversations2retrieval.py) to convert the conversation into input to send the retriever. You can achieve this anyway you desire such as a rewrite of the conversation, using only the last question, trying to generate an answer etc.. 

[Conversations](/human/conversations/conversations.json)  
[Corpora](/corpora/)  
[Sample Data](/human/retrieval_tasks/)  

Your final output should be a tsv file that includes the query-id and corpus-id for all relevant passages per question.

### Output Format:

```
query-id \t corpus-id \t score
```

The score will not be used, so it can always be 1. The corpus-id is per line. If the query has multiple relevant passages, they should be on separate lines. Sample output can be seen in the Sample Data folder above.

## Task B: Generation with Reference Passages

[Data](/human/generation_tasks/reference.jsonl)

## Task C: Full RAG

[Data](/human/generation_tasks/reference+RAG.jsonl)

## Evaluation Scripts -->

## Task Organizers

Sara Rosenthal [✉️](mailto:sjrosenthal@us.ibm.com)  
Yannis Katsis [✉️](mailto:yannis.katsis@ibm.com)  
Vraj Shah [✉️](mailto:vraj@ibm.com)  
Marina Danilevsky [✉️](mailto:mdanile@us.ibm.com)   
