import os
import re
import fire
import pandas as pd
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
import warnings


def prepare_prompts(
    tokenizer, 
    df, 
    prompt_template_file, 
    unsafety_instructions_file
):
    scenario2instructions = pd.read_csv(unsafety_instructions_file)\
        .set_index("scenario")["instructions"].to_dict()
    
    with open(prompt_template_file, 'r') as f:
        prompt_template = f.read()

    prompts = [
        prompt_template.format(
            SCENARIO=row["scenario"], 
            SCENARIO_INSTRUCTIONS=scenario2instructions[row["scenario"]], 
            PROMPT=row["prompt"], 
            RESPONSE=row["response"]
        ) for idx, row in df.iterrows()
    ]

    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], 
            tokenize=False, 
            add_generation_prompt=True
        ) for prompt in prompts
    ]
    
    return chats


def parse_llm_response(response):
    # Check if the response contains the expected "Answer:" line
    match = re.search(r"Answer:\s*(safe|unsafe|ambiguous|error)", response, re.IGNORECASE)
    
    if match:
        result = match.group(1).lower()
        return result
    else:
        return None
    

def save_results(savepath, df):
    
    df.to_csv(savepath, index=False)
    print(f"Results saved in {savepath}")


def run(
    datapath: str,
    savepath: str, 
    prompt_template_file: str,
    unsafety_instructions_file: str = "/root/llm_quant_safety/evaluate/alignment/eval_instructions.csv",
    model_name: str = "google/gemma-2-27b-it",
    hf_cache_path: str = "/home/data/v.moskvoretskii/cache/",
    max_tokens: str = 256
):
    warnings.filterwarnings("ignore")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print('Preparing prompts')
    df = pd.read_csv(datapath)
    formatted_prompts = prepare_prompts(tokenizer, df, prompt_template_file, unsafety_instructions_file)
    
    llm = LLM(
        model=model_name, 
        dtype=torch.bfloat16, 
        trust_remote_code=True,
        tensor_parallel_size=2,
        download_dir=hf_cache_path
    )

    sampling_params = SamplingParams(
        temperature=0.6, 
        top_p=0.9, 
        max_tokens=max_tokens
    )

    outputs = llm.generate(
        list(formatted_prompts), 
        sampling_params
    )
    llm_evaluations_responses = [output.outputs[0].text for output in outputs]
    df["llm_reasoning"] = llm_evaluations_responses
    df["llm_eval"] = [parse_llm_response(response) for response in llm_evaluations_responses]

    df.to_csv(savepath, index=False)
    print(f"Results saved in {savepath}")

if __name__ == "__main__":
    fire.Fire(run)

