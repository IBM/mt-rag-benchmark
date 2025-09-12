import argparse
import os
from judge_wrapper import *
from run_algorithmic import run_algorithmic_judges

def args_parser():
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
        "--algorithmic_evaluators",
        type=str,
        dest="evaluators",
        help="Algorithmic Evaluators configuration file",
        default="scripts/evaluation/config.yaml",
    )
    parser.add_argument(
        "--provider", 
        type=str,
        required=True,
        dest="provider", 
        choices=["openai", "hf"],
        help="Provider to use for LLM judges",
    )
    parser.add_argument(
        "--judge_model", 
        type=str, 
        dest="judge_model",
        help="Hugging Face model name (required if provider=hf)"
    )
    parser.add_argument(
        "--openai_key", 
        type=str, 
        help="OpenAI Key (required if provider=openai)"
    )
    parser.add_argument(
        "--azure_host", 
        type=str, 
        help="OpenAI endpoint (required if provider=openai)"
    )
    return parser

if __name__ == "__main__":
    parser = args_parser()
    args = parser.parse_args()
    
    run_algorithmic_judges(args.evaluators, args.input, args.output)
    
    if args.provider == "openai":
        if not args.openai_key or not args.azure_host:
            parser.error("--provider openai requires --openai_key and --azure_host")
            
        os.environ["AZURE_OPENAI_API_KEY"] = args.openai_key
        os.environ["OPENAI_AZURE_HOST"] = args.azure_host
    else:
        if not args.judge_model:
            parser.error(f"--provider {args.provider} requires --judge_model")

        judge_model = args.judge_model
    
    
    if args.provider == "openai":
        run_idk_judge(args.provider, args.output, args.output)
        run_ragas_judges_openai(args.output, args.output, args.openai_key, args.azure_host)
        run_radbench_judge(args.provider, args.output, args.output)
        
        get_idk_conditioned_metrics(args.output, args.output)
    else:
        run_idk_judge(args.judge_model, args.output, args.output)
        run_ragas_judges_local(judge_model, args.output, args.output)
        run_radbench_judge(judge_model, args.output, args.output)
        
        get_idk_conditioned_metrics(args.output, args.output)