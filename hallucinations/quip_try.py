import fire
from transformers import AutoTokenizer

from quipsharp.quantizer import load_quantized_model

def run(prompt: str = "What is HTML"):
    model_path = "/root/llm_quant_safety/quantization/models/Llama-3.1-8B-Instruct-quip-2bit"
    quant_model = load_quantized_model(model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').cuda()
    
    print(tokenizer.decode(quant_model.generate(encoded, do_sample=True, max_new_tokens=128)[0]))

if __name__ == "__main__":
    fire.Fire(run)