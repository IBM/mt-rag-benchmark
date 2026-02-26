# Generation Tasks

 The conversations are converted into 842 tasks. A task is a conversation turn containing all previous turns together with the last user question (e.g., the task created for turn $k$ includes all user and agent questions/responses for the first $k-1$ turns plus the user question for turn $k$). Our generation tasks measure performance under three retrieval settings.

| Setting  | Description | File | # Tasks
| ------------- | ------------- |  ------------- |  ------------- | 
| Reference  | Generation using reference passages | [reference.jsonl](reference.jsonl) |  842 | 
| Reference + RAG | Retrieval followed by generation but with the reference passages kept in the top 5 passages. (Restricted to tasks that have 2 contexts or less) | [reference+RAG.jsonl](reference+RAG.jsonl) | 436 |
| Full RAG | Retrieval followed by generation where retrieval results consist of the top 5 passages. | [RAG.jsonl](RAG.jsonl) | 842 |

Generation experiments can be run using any desired models e.g. available on HuggingFace.

The format of a task in the files is as follows:

```
{
  "task_id": "unique_id_as_string",
  "conversation_id": "unique_id_as_string",
  "task_type": "rag",
  "turn": (int) position of current turn in conversation
  "collection": "name of domain specific database",
  "dataset": "MT-RAG Authors (Internal)",
  "contexts": [
    {
      "document_id": "id_1",
      "text": "document text for model consumption",
      "title": "string [optional]"
      "url": "string [optional]",
      "score": (float) passage_score,
      "feedback": {
        "relevant": {
          "author_XYZ: {
            "value": "yes|no",
            "timestamp": unix timestamp
          }
        }
      },
      "query": { query metadata and string },
    },
    {
      "document_id": "id_2",
      "text": "document text for model consumption",
      "title": "string [optional]"
      "url": "string [optional]",
      "score": (float) passage_score,
      "feedback": {
        "relevant": {
          "author_XYZ: {
            "value": "yes|no",
            "timestamp": unix timestamp
          }
        }
      },
      "query": { query metadata and string },
    }
  ],
  "input": [
    { ... additional turns if applicable ... }
    {
      "speaker": "user",
      "text": "query text",
      "metadata": {
        "author_type": "human",
        "author_id": "author_XYZ",
        "created_at": unix timestamp
      },
      "enrichments": {
        "answerability": [
          "ANSWERABLE|UNANSERABLE|CONVERSATIONAL|PARTIAL"
        ],
        "Question Type": [
          "Factoid|Composite|Keyword|Opinion|..."
        ],
        "Multi-Turn": [
          "Follow-up|Clarification"
        ]
      }
    }
  ],
  "targets": [
    {
      "speaker": "agent",
      "text": "response text"
      "metadata": {
        "author_type": "model",
        "author_id": "mixtral-8x7b-instruct-v01",
        "created_at": unit timestamp
      }
    }
  ],
"answerability": [
    "ANSWERABLE|UNANSERABLE|CONVERSATIONAL|PARTIAL"
  ],
  "Question Type": [
    "Factoid|Composite|Keyword|Opinion|..."
  ],
  "Multi-Turn": [
    "Follow-up|Clarification"
  ]
}
```
