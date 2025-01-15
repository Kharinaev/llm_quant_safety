import os
import sys
import fire
import shutil
import pandas as pd
import torch
from transformers import AutoTokenizer, GenerationConfig
import warnings
from tqdm import tqdm

from quipsharp.quantizer import load_quantized_model

import warnings
warnings.filterwarnings('ignore')


RESULTS_COL = "llm_responses"


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
        )[:-len(tokenizer.eos_token)] for prompt in prompts
    ]

    return chats

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
                pad_to_multiple_of=8,
                add_special_tokens=False
            ).to("cuda") 
        )
    tokenizer.padding_side="right"
    return batches_tok


def init_data(df: pd.DataFrame, bs: int, tokenizer) -> int:
    """
    Function inits torch dataloader only for undone examples
    """
    # if RESULTS_COL not in df.columns:
    #     df[RESULTS_COL] = None
    # list_results = list(df[RESULTS_COL])

    # count_done = next(
    #     (i for i, val in enumerate(list_results) if not isinstance(val, str)),
    #     len(list_results)  # Если все элементы строки, начать с конца
    # )
    
    # df_undone = df.iloc[count_done:]
    # chats = prepare_prompts(tokenizer, df_undone)
    chats = prepare_prompts(tokenizer, df)
    prompt_batches = batch_prompts(chats, tokenizer, bs)
    return prompt_batches


def run_model(
    model_name: str, 
    df: pd.DataFrame, 
    batch_size: int, 
    results: list,
    max_tokens: int,
):
    """
    Runs a model, saving outputs to `results` list 
    """
    with torch.no_grad():
    
        model = load_quantized_model(model_name, device_map='auto')
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        model.eval()
    
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        gen_conf = GenerationConfig(
            do_sample=False,
            max_new_tokens=max_tokens,
            temperature=0,
            pad_token_id=tokenizer.pad_token_id
        )
        
        prompt_batches = init_data(df, batch_size, tokenizer)
    
        for prompts_tokenized in tqdm(prompt_batches):
            outputs_tokenized = model.generate(**prompts_tokenized, generation_config=gen_conf)
        
            outputs_tokenized = [ 
                tok_out[len(tok_in):] for tok_in, tok_out 
                in zip(prompts_tokenized["input_ids"], outputs_tokenized) 
            ] 
        
            outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
            results.extend(outputs)

            del prompts_tokenized
            del outputs_tokenized
            torch.cuda.empty_cache()


def save_results(df, results, savepath):
    # list_results = list(df[RESULTS_COL])
    # try:
    #     first_none_idx = next(i for i,val in enumerate(list_results) if not isinstance(val,str))
    #     for i, elem in enumerate(results):
    #         list_results[first_none_idx+i] = elem
    
    #     df[RESULTS_COL] = list_results
    #     df.to_csv(savepath, index=False)
    #     print(f"Results saved in {savepath}")
        
    # except Exception as e:
    #     print(f"Error saving results: {str(e)}")
    df[RESULTS_COL] = results
    df.to_csv(savepath, index=False)
    print(f"Results saved in {savepath}")

# def check_and_copy(datapath, savepath):
#     """
#     Проверяет существование файла по savepath. Если его нет, копирует файл из datapath в savepath.
    
#     :param datapath: Путь к исходному файлу
#     :param savepath: Путь, куда нужно сохранить файл
#     """
#     if not os.path.exists(savepath):
#         # Создание папок, если они отсутствуют
#         os.makedirs(os.path.dirname(savepath), exist_ok=True)
#         shutil.copy(datapath, savepath)
#         print(f"Файл скопирован из '{datapath}' в '{savepath}'.")
#     else:
#         print(f"Файл уже существует по пути '{savepath}'.")


def run(
    datapath: str,
    savepath: str,
    model_name: str = "/root/llm_quant_safety/quantization/models/Llama-3.1-8B-Instruct-quip-2bit",
    max_tokens: int = 256,
    batch_size: int = 32,
):
    
    # check_and_copy(datapath, savepath)
    
    results = []   
    df = pd.read_csv(datapath)

    # try:
    #     run_model(model_name, df, batch_size, results, max_tokens)
    # except Exception as e:
    #     print(str(e))
    # finally:
    #     save_results(df, results, savepath)
    #     print(len(results))  # Выводим количество обработанных примеров
    #     sys.exit(0)
    run_model(model_name, df, batch_size, results, max_tokens)
    save_results(df, results, savepath)

if __name__ == "__main__":
    fire.Fire(run)