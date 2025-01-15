import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import pandas as pd
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm.auto import tqdm

model_name = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
tokenizer = AutoTokenizer.from_pretrained(model_name)

llm = LLM(
    model=model_name, 
    dtype=torch.bfloat16, 
    trust_remote_code=True,
    enforce_eager=True,
)

df = pd.read_csv('data/multilingual_safety_benchmark.csv')
df = df[df.lang=='en'].copy()
print(df.info())

lang_instruction = {    
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


def generate_safety_check_prompt(prompt, lang):
    user_message = lang_instruction[lang] + '\n' + prompt
    messages = [{"role": "user", "content": user_message}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return formatted

formatted_prompts = df.apply(
    lambda row: generate_safety_check_prompt(row.prompt, row.lang), 
    axis=1
)

sampling_params = SamplingParams(
    temperature=0.6, 
    top_p=0.9, 
    max_tokens=512
)

outputs = llm.generate(
    list(formatted_prompts), 
    sampling_params
)

llm_answers = [output.outputs[0].text for output in outputs]

df['response'] = llm_answers
df['model'] = 'llama_abliterated'
# df.to_csv('lama_AQLM_final.csv')
df.to_csv('data/en_responses_abliterated.csv', index=False)

