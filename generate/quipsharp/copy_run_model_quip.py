import argparse
import sys
import os
import gc

import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
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

class CustomDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return text

def init_dataloader(df: pd.DataFrame, bs: int) -> int:
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
        formatted_prompts.append(messages)

    # init ds and dl
    collate = lambda x: x
    ds = CustomDataset(formatted_prompts)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate)
    return dl

def run_model(model_name: str, df: pd.DataFrame, bs: int, results: list):
    """
    Runs a model, saving outputs to `results` list 
    """
    torch.no_grad()
    
    dl = init_dataloader(df, bs)

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_quantized_model(model_name, device_map='auto')
    # model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    model.eval()

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=model_name,
        device_map="auto",
        batch_size=bs,
    )
    pipeline.tokenizer.add_special_tokens({'pad_token': '<|eot_id|>'}) #LLAMA
    pipeline.tokenizer.padding_side = 'left'
    pipeline.model.generation_config.pad_token_id = pipeline.tokenizer.pad_token_id
    

    for batch_prompts in tqdm(dl):
        
        batch_results = pipeline(
            batch_prompts,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
        
        answers = [
            result["generated_text"][-1]["content"]
            for result in batch_results
        ]
        results.extend(answers)
        
        del batch_prompts
        torch.cuda.empty_cache()
        gc.collect()
    

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

    os.environ["CUDA_VISIBLE_DEVICES"]=f"{par}"
    
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