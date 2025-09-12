import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class HuggingFaceLLMClient:
    def __init__(self, model_name, model_dir=None, use_4bit=False):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                          cache_dir=model_dir,
                                                          torch_dtype=torch.float16,
                                                          device_map="auto",
                                                          load_in_4bit=True,
                                                          )
        self.model.eval()

    def generate_response(self, user_input, max_new_tokens=1024, temperature=0, top_p=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "do_sample": False 
        }
        
        prompt = user_input

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **_params
            )
        
        new_tokens = output_ids[0][input_length:]
        
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response


if __name__ == "__main__":

    model_name = "mistralai/Mixtral-8x22B-Instruct-v0.1"    
    model_dir = 'models'
    
    print(f"Running model {model_name}")
    
    client = HuggingFaceLLMClient(model_name, model_dir = model_dir)
    user_input = '\n[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question given the provided document and a reference answer.\n\nYour evaluation should assess the faithfulness, appropriateness, and completeness. Your evaluation should focus on the assistant\'s answer to the question of the current turn.  You will be given the assistant\'s answer and a reference answer. You will also be given the user questions and assistant\'s answers of the previous turns of the conversation. You should consider how well the assistant\'s answer captures the key information, knowledge points mentioned in the reference answer, and how it respects or builds upon the focus and knowledge points from the previous turns. \n\n[Faithfulness]: You are given the full conversation, the question of the current turn, the assistant\'s answer, and documents. You should evaluate how faithful is the assistant\'s answer to the information in the document and previous conversation.\n[Appropriateness]: You should evaluate if the assistant\'s answer is relevant to the question of the current turn and if it addresses all the issues raised by the question without adding extra information.\n[Completeness]: You should evaluate whether the assistant\'s answer is complete with information from the documents.\n\nBegin your evaluation by comparing the assistant\'s answer against the reference answer in this turn. Be as objective as possible, and provide a detailed justification for your rating. After providing your explanation, you must rate the response on a scale of 1 to 10, strictly following this format: "Rating: [[rating]]", for example: "Rating: [[5]]".\n\n[The Start of Previous Conversation]\nuser: where do the arizona cardinals play this week\nagent: I\'m sorry, but I don\'t have the answer to your question.\n[The End of Previous Conversation]\n\n[The Start of Current Turn Question]\nDo the Arizona Cardinals play outside the US?\n[The End of Current Turn Question]\n\n[The Start of Reference Answer]\nThe Arizona Cardinals do play outside the United States. They had a game in London, England, on October 22, 2017, against the Los Angeles Rams at Twickenham Stadium and in 2005 they played in Mexico.\n[The End of Reference Answer]\n\n[The Start of Assistant\'s Answer]\nYes, the Arizona Cardinals played one road game outside the US in the 2017 season. They played the Los Angeles Rams at Twickenham Stadium in London, England on October 22, 2017, as part of the NFL International Series.\n[The End of Assistant\'s Answer]\n\n[The Start of Document]\n2017 Arizona Cardinals season\nWeek Date Opponent Result Record Game site NFL.com recap September 10 at Detroit Lions L 23 -- 35 0 -- 1 Ford Field Recap September 17 at Indianapolis Colts W 16 -- 13 ( OT ) 1 -- 1 Lucas Oil Stadium Recap September 25 Dallas Cowboys L 17 -- 28 1 -- 2 University of Phoenix Stadium Recap October 1 San Francisco 49ers W 18 -- 15 ( OT ) 2 -- 2 University of Phoenix Stadium Recap 5 October 8 at Philadelphia Eagles L 7 -- 34 2 -- 3 Lincoln Financial Field Recap 6 October 15 Tampa Bay Buccaneers W 38 -- 33 3 -- 3 University of Phoenix Stadium Recap 7 October 22 at Los Angeles Rams L 0 -- 33 3 -- 4 Twickenham Stadium ( London , England ) Recap 8 Bye 9 November 5 at San Francisco 49ers W 20 -- 10 4 -- 4 Levi \'s Stadium Recap 10 November 9 Seattle Seahawks L 16 -- 22 4 -- 5 University of Phoenix Stadium Recap 11 November 19 at Houston Texans L 21 -- 31 4 -- 6 NRG Stadium Recap 12 November 26 Jacksonville Jaguars W 27 -- 24 5 -- 6 University of Phoenix Stadium Recap 13 December 3 Los Angeles Rams L 16 -- 32 5 -- 7 University of Phoenix Stadium Recap 14 December 10 Tennessee Titans W 12 -- 7 6 -- 7 University of Phoenix Stadium Recap 15 December 17 at Washington Redskins L 15 -- 20 6 -- 8 FedExField Recap 16 December 24 New York Giants W 23 -- 0 7 -- 8 University of Phoenix Stadium Recap 17 December 31 at Seattle Seahawks W 26 -- 24 8 -- 8 CenturyLink Field Recap\n\n2017 Arizona Cardinals season\nOn December 13 , 2016 , the NFL announced that the Cardinals will play the Los Angeles Rams as one of the NFL International Series at Twickenham Stadium in London , England , with the Rams serving as the home team . It will be the Cardinals \' first appearance in the International Series , though the Cardinals played in Mexico in 2005 against the San Francisco 49ers . The game will occur during Week 7 on Sunday , October 22 , and will be televised in the United States . The network and date were announced in conjunction with the release of the 2017 regular season schedule .\n\n2017 Arizona Cardinals season\nThe 2017 Arizona Cardinals season was the franchise \'s 98th season in the National Football League , the 30th in Arizona and 12th at University of Phoenix Stadium . It was also the fifth and final season under head coach Bruce Arians . The Cardinals played one road game in London at Twickenham Stadium against the Los Angeles Rams as one of the NFL London Games . They improved on a 7 -- 8 -- 1 season they had in 2016 , finishing 8 -- 8 . However , they missed the playoffs for the second straight season .\n[The End of Document]\n'

    response = client.generate_response(user_input)
    print("Model Response:", response)
