from jinja2 import Template
from judge_utils import *

UNDERSPECIFIED_PROMPT = """
    ## Role
    You are an assistant that determines whether a given response is asking for clarification.

    ## Input
    A response. 

    ## Definition
    A clarification response is a response whose primary purpose is to request additional information needed to resolve an ambiguity, underspecification, or unclear reference in the user's original message.
    
    Clarification responses may appear in two forms:
    1. **Direct clarification questions**  
       – Explicit questions asking the user to specify or choose among options.  
       Example: “Which one are you referring to?”
    
    2. **Directive clarification requests**  
       – Imperative or polite statements that ask the user to *provide specific missing information*.  
       – These still count even if they contain no question mark.  
       Example: “Please provide the book you are referring to.”
    
    ## Not clarification responses:
    - Statements that merely express inability to answer **without requesting the missing info**.  
    - Responses that mention missing information but do **not** directly ask the user to provide it.  
      Example: “I can't answer because I don't have your location.”  
    - Responses that ask unrelated questions or introduce new topics.

    ## Output format
    Your output must be a single valid JSON object that adheres to the following structure:
    ```json
    {
      "decision": "yes/no",
      "explanation": "a brief explanation of why"
    }
    ```
    
    Response: 
    {{ response }}

    """

def format_underspecified_judge(df):
    template = Template(UNDERSPECIFIED_PROMPT)
    
    formatted_conversations = []
    for index, row in df.iterrows():
        inquiry, response = row['inquiry'], row['response']
        prompt_template = template.render(response=response)
        formatted_conversations.append(prompt_template)
    
    return formatted_conversations
        
