import argparse
import sys
import os
import gc

import torch
import transformers
from transformers import AutoTokenizer, GenerationConfig
from tqdm import tqdm
import pandas as pd

from quantizer import load_quantized_model

import warnings
warnings.filterwarnings('ignore')

RESULTS_COL = 'response'

LANG_INSTRUCTIONS = {    
    "ru": "Отвечай на Русском языке",
    "bn": "বাংলা ভাষায় উত্তর",    
    "de": "Antworte auf Deutsch",
    "en": "Answer in English",    
    "fr": "Répondez en français",
    "hi": "उत्तर हिंदी में",    
    "ja": "日本語での回答",
    "ar": "الإجابة باللغة العربية",    
    "sp": "Contesta en español",
    "zh": "中文答案"
}

def prepare_prompts(prompts, tokenizer, batch_size=16):
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
    # count undone
    list_results = list(df[RESULTS_COL])
    count_done = next(i for i,val in enumerate(list_results) if not isinstance(val,str))
    df_undone = df.iloc[count_done:].copy()
    
    # format prompts into chat template
    formatted_prompts = []
    for idx, row in df_undone.iterrows():
        user_message = LANG_INSTRUCTIONS[row.lang] + '\n' + row.prompt
        messages = [{"role": "user", "content": user_message}]
        messages = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(messages)

    prompt_batches = prepare_prompts(formatted_prompts, tokenizer, batch_size=bs)
    return prompt_batches


def run_model(model_name: str, df: pd.DataFrame, bs: int, results: list):
    """
    Runs a model, saving outputs to `results` list 
    """
    with torch.no_grad()
    
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = load_quantized_model(model_name, device_map='auto')
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        model.eval()
    
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    
        gen_conf = GenerationConfig(
            do_sample=True,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
        
        prompt_batches = init_data(df, bs, tokenizer)
    
        for prompts_tokenized in tqdm(prompt_batches):
            outputs_tokenized = model.generate(**prompts_tokenized, generation_config=gen_conf)
        
            # remove prompt from gen. tokens
            outputs_tokenized = [ 
                tok_out[len(tok_in):] for tok_in, tok_out 
                in zip(prompts_tokenized["input_ids"], outputs_tokenized) 
            ] 
        
            # count and decode gen. tokens 
            num_tokens = sum([len(t) for t in outputs_tokenized])
            outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
        
            # store in results to be gathered by accelerate
            results.extend(outputs)
    

def save_results(df, results, datapath):
    list_results = list(df[RESULTS_COL])
    first_none_idx = next(i for i,val in enumerate(list_results) if not isinstance(val,str))
    for i, elem in enumerate(results):
        list_results[first_none_idx+i] = elem

    df[RESULTS_COL] = list_results
    df.to_csv(datapath, index=False)
    

def main():
    parser = argparse.ArgumentParser(description="Запуск модели с заданными параметрами")
    parser.add_argument("--model-name", type=str, required=True, help="Название модели")
    parser.add_argument("--datapath", type=str, required=True, help="Путь к данным")
    parser.add_argument("--bs", type=int, required=True, help="Размер батча")
    parser.add_argument("--gpu-id", type=int, required=True, help="Номер GPU")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpu_id}"
    
    results = []   
    df = pd.read_csv(args.datapath)
    try:
        run_model(args.model_name, df, args.bs, results)
    except Exception as e:
        # print(f"Error occurred: {str(e)}", file=sys.stderr)
        print(str(e))
    finally:
        save_results(df, results, args.datapath)
        print(len(results))  # Выводим количество обработанных примеров
        sys.exit(0)

if __name__ == "__main__":
    main()