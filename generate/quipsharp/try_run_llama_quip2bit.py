import os, argparse
from transformers import AutoTokenizer

from quantizer import load_quantized_model

def run_model(prompt):
    model_path = "/root/llm_quant_safety/quantization/models/Llama-3.1-8B-Instruct-quip-2bit"
    quant_model = load_quantized_model(model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').cuda()
    
    print(tokenizer.decode(quant_model.generate(encoded, do_sample=True, max_new_tokens=128)[0]))

def main():
    parser = argparse.ArgumentParser(description="Запуск модели с заданными параметрами")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--gpu-id", type=str, default="0")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpu_id}"
    
    run_model(args.prompt)

if __name__ == "__main__":
    main()