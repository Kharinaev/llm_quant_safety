import os
import re
import math
import fire
import pandas as pd
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import warnings


def get_most_probable_option_by_logits(
    vllm_logits_dict,
    num_options,
    tokenizer
):
    probs = {}
    for opt in range(num_options):
        token_id = tokenizer.convert_tokens_to_ids(str(opt))
        if token_id in vllm_logits_dict:
            logit_obj = vllm_logits_dict[token_id]
            probs[opt] = math.exp(logit_obj.logprob)

    if len(probs) > 0:
        chosen_opt = max(probs, key=probs.get)
        return {
            "option" : chosen_opt,
            "prob" : probs[chosen_opt]
        }
    else:
        return {
            "option" : None,
            "prob" : None
        }

def process_responses(
    all_outputs,
    df,
    tokenizer
):
    results = [
        get_most_probable_option_by_logits(
            all_outputs[idx].outputs[0].logprobs[0],
            len(eval(df.options.iloc[idx])),
            tokenizer
        )
        for idx in range(len(df))
    ]
    return results

def run(
    savepath: str, 
    model_name: str = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
    datapath: str = '/root/llm_quant_safety/data/safetybench/test_en_5shot_assistant_prefill.csv',
    hf_cache_path: str = "/home/data/v.moskvoretskii/cache/",
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
    if 'llama' in model_name.lower():
        prompts = df['llama_prompt'].tolist()
    elif 'mistral' in model_name.lower():
        prompts = df['mistral_prompt'].tolist()
        

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
        logprobs=20,
        max_tokens=1,
        temperature=1
    )

    outputs = llm.generate(
        prompts, 
        sampling_params
    )

    responses_dicts = process_responses(outputs, df, tokenizer)

    df_res = pd.concat(
        [
            df,
            pd.DataFrame(responses_dicts, index=df.index),
        ],
        axis=1
    )
    df_res.to_csv(savepath, index=False)
    print(f"Results saved in {savepath}")

    json_path = savepath[:-4] + '.json'
    df_res.set_index('id')['option'].to_json(json_path)
    print(f"JSON Results saved in {json_path}")

    

if __name__ == "__main__":
    fire.Fire(run)

