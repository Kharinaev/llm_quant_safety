import os
import re
import fire
import pandas as pd
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
import warnings


PROMPT_TEMPLATE = """
You are helpful AI assistant.
Briefly answer the question based on provided info.

# Information
{INFO}

# Question
{QUESTION}
""".strip()


def prepare_prompts(
    tokenizer, 
    df
):
    prompts = df.prompt

    chats = [
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt}, 
                {"role": "assistant", "content" : "Answer: "}
            ],
            tokenize=False, 
            add_generation_prompt=True
        ) for prompt in prompts
    ]
    
    return chats
    

def save_results(savepath, df):
    
    df.to_csv(savepath, index=False)
    print(f"Results saved in {savepath}")


def run(
    datapath: str,
    savepath: str, 
    model_name: str = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
    hf_cache_path: str = "/home/data/v.moskvoretskii/cache/",
    max_tokens: str = 8,
    tokenizer_name: str = None,
    model_dtype = "bfloat16",
    gpu_memory_utilization : float = 0.9,
):
    warnings.filterwarnings("ignore")
    if not tokenizer_name:
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print('Preparing prompts')
    df = pd.read_csv(datapath)
    formatted_prompts = prepare_prompts(tokenizer, df)

    if model_dtype == "bfloat16":
        model_dtype = torch.bfloat16
    elif model_dtype == "float16":
        model_dtype = torch.float16
    else:
        raise ValueError(f"dtype {model_dtype} not implemented")
    
    llm = LLM(
        model=model_name, 
        dtype=model_dtype, 
        trust_remote_code=True,
        # tensor_parallel_size=2,
        download_dir=hf_cache_path,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        temperature=0, 
        max_tokens=max_tokens
    )

    outputs = llm.generate(
        list(formatted_prompts), 
        sampling_params
    )
    llm_responses = [output.outputs[0].text for output in outputs]
    df["llm_responses"] = llm_responses

    df.to_csv(savepath, index=False)
    print(f"Results saved in {savepath}")

if __name__ == "__main__":
    fire.Fire(run)

