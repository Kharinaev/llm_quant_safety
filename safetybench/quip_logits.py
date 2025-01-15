import fire
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from quipsharp.quantizer import load_quantized_model

import warnings
warnings.filterwarnings('ignore')


def batch_prompts(prompts, tokenizer, batch_size=16):
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok=[]
    tokenizer.padding_side="left"     
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                padding='longest', 
                truncation=False, 
                pad_to_multiple_of=2,
                add_special_tokens=False
            ).to("cuda") 
        )
    tokenizer.padding_side="right"
    return batches_tok


def init_data(prompts: list, bs: int, tokenizer) -> int:
    """
    Function inits torch dataloader only for undone examples
    """
    # sorted_by_len = prompts.sort(reverse=True, key=lambda pr: len(pr))
    # long_prompts = sorted_by_len_prompts[:num_first]
    # short_prompts = sorted_by_len_prompts[num_first:]

    # long_prompts_batches = batch_prompts(long_prompts, tokenizer, num_first)
    # short_prompt_batches = batch_prompts(short_prompts, tokenizer, bs)

    # sorted_batches = long_prompts_batches + short_prompt_batches
    
    # return sorted_batches

    prompt_batches = batch_prompts(prompts, tokenizer, bs)
    return prompt_batches


def get_most_probable_option_by_logits_hf(
    probabilities,  # from torch.topk 
    tokens,  # from torch.topk 
    num_options,
    tokenizer
):
    probabilities, tokens = probabilities.cpu(), tokens.cpu()
    probs_dict = {token.item(): prob.item() for prob, token in zip(probabilities, tokens)}
    
    probs = {}
    
    for opt in range(num_options):
        token_id = tokenizer.convert_tokens_to_ids(str(opt))
        if token_id in probs_dict:
            probs[opt] = probs_dict[token_id]

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


def topk_batch(
    logits,  # [batch_size, max_seq_len, vocab_size]
    attention_mask, # [batch_size, max_seq_len]
    k
):
    # seq_lengths = attention_mask.sum(-1) - 1  # [batch_size]
    # last_token_probs = torch.take_along_dim(
    #     logits, 
    #     seq_lengths[:, None, None], 
    #     1
    # )[:, 0, :].softmax(-1)  # [batch_size, vocab_size]
    # last_token_probs = logits[:, seq_lengths][:, 0, :].softmax(-1)
    # last_token_probs = logits[torch.arange(logits.size(0)), seq_lengths].softmax(-1)
    last_token_probs = logits[:, -1, :].softmax(-1)

    probs, tokens = torch.topk(last_token_probs, k=k, dim=-1)
    return probs.cpu(), tokens.cpu()


def run_model(
    model_name: str, 
    tokenizer,
    prompts: list, 
    batch_size: int, 
):
    """
    Runs a model, saving outputs to `results` list 
    """
    
    with torch.no_grad():
    
        model = load_quantized_model(model_name, device_map='auto')
        # model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        model.eval()
    
        
        prompt_batches = init_data(prompts, batch_size, tokenizer)

        all_topk_probs = []
        all_topk_tokens = []
        
        for encoded in tqdm(prompt_batches):
            
            outputs = model.forward(encoded['input_ids'])
        
            batch_probs, batch_tokens = topk_batch(
                outputs.logits,
                encoded.attention_mask,
                20 # 20
            )

            all_topk_probs.append(batch_probs)
            all_topk_tokens.append(batch_tokens)
            
            del encoded
            del outputs
            # torch.cuda.empty_cache()

        all_topk_probs_tensor = torch.cat(
            all_topk_probs,
            dim=0
        )
        all_topk_tokens_tensor = torch.cat(
            all_topk_tokens,
            dim=0
        )

        print(f'\nTensors shapes: probs - {all_topk_probs_tensor.shape}, tokens - {all_topk_tokens_tensor.shape}\n')
    return all_topk_probs_tensor, all_topk_tokens_tensor


def process_responses(
    all_probas, 
    all_tokens, 
    df, 
    tokenizer
):
    results = [
        get_most_probable_option_by_logits_hf(
            all_probas[idx, :],
            all_tokens[idx, :],
            len(eval(df.options.iloc[idx])),
            tokenizer
        )
        for idx in range(len(df))
    ]
    return results
    

def run(
    datapath: str,
    savepath: str,
    model_name: str = '/home/data/v.moskvoretskii/quip/models/Llama-3.1-8B-Instruct-quip-2bit',
    batch_size: int = 32,
):   
    df = pd.read_csv(datapath)

    if 'llama' in model_name.lower():
        # df = df.sort_values(
        #     'llama_prompt', 
        #     key=lambda x: x.str.len(),
        #     ascending=False
        # )
        prompts = df['llama_prompt'].tolist()
    elif 'mistral' in model_name.lower():
        # df = df.sort_values(
        #     'mistral_prompt', 
        #     key=lambda x: x.str.len(),
        #     ascending=False
        # )
        prompts = df['mistral_prompt'].tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    all_probas, all_tokens = run_model(model_name, tokenizer, prompts, batch_size)
    responses_dicts = process_responses(all_probas, all_tokens, df, tokenizer)
    df_res = pd.concat(
        [
            df,
            pd.DataFrame(responses_dicts, index=df.index),
        ],
        axis=1
    )
    df_res.to_csv(savepath, index=False)
    print(f"Results saved in {savepath}")


if __name__ == "__main__":
    fire.Fire(run)