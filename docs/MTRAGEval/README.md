# MTRAGEval

🎉 Welcome to MTRAGEval! MTRAGEval is a task for Evaluating Multi-Turn RAG Conversations at [SemEval 2026](https://semeval.github.io/SemEval2026/). 🎉

## 📝 Registration

Please fill out this form to register for our task if you plan to participate in MTRAGEval: [Registration Form](https://forms.gle/XgjqgDGgG7wgCb5r6)

## Join Our Mailing List!

[MTRAGEval Mailing List](https://groups.google.com/g/mtrageval)

## Training and Trial Data

The MTRAG Benchmark is released as the trial and training data for MTRAG. You can access the full dataset [here](https://github.com/IBM/mt-rag-benchmark/)

Note: The MTRAG Benchmark includes metadata that describes dimensions for each task including the question type (e.g. factoid), answerability (e.g. unanswerable, answerable), and multi-turn type (e.g. follow-up, clarification). This information will *NOT* be provided during evaluation. We will only provide the corpus domain e.g. ClapNQ, Govt.

## 📋 Tasks

* Task A: Retrieval Only
* Task B: Generation with Reference Passages (Reference)
* Task C: Generation with Retrieved Passages (RAG)

Read more about our tasks in our [proposal](../MT_RAG_SemEval_Proposal.pdf)

## Evaluation 

### Evaluation Scripts

Retrieval and Generation Evaluation Scripts are available on the GitHub repo! Please visit the evaluation [README](https://github.com/IBM/mt-rag-benchmark/blob/main/scripts/evaluation/README.md) for more information.

⌛ Coming Soon: Validation Script

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

The evaluation data will be provided to all registered participants at the start of each evaluation phase.

Note: The MTRAG Benchmark includes metadata that describes dimensions for each task including the question type (e.g. factoid), answerability (e.g. unanswerable, answerable), and multi-turn type (e.g. follow-up, clarification). This information will NOT be provided during evaluation. We will only provide the corpus domain e.g. ClapNQ, Govt.

## 📆 Timeline (Tentative)

* Sample and Training data ready 15 July 2025
* Evaluation start 10 January 2026 Task A and C
* Evaluation end 20 January 2026 Task A and C
* Evaluation start 21 January 2026 Task B
* Evaluation end by 31 January 2026 Task B
* Paper submission due February 2026
* Notification to authors March 2026
* Camera ready due April 2026
* SemEval workshop Summer 2026 (co-located with a major NLP conference)


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

Sara Rosenthal [✉️](sjrosenthal@us.ibm.com)  
Yannis Katsis [✉️](yannis.katsis@ibm.com)  
Vraj Shah [✉️](vraj@ibm.com)  
Marina Danilevsky [✉️](mdanile@us.ibm.com)   
