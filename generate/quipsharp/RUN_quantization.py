import argparse, sys, os
from pathlib import Path
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantizer import QuipQuantizer

def setup_logging():
    # Настройка корневого логгера
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # терминал
            logging.FileHandler('logs/llama_e8p12.log')
        ]
    )

def run_quantization(model_name, quant_dir, codebook):

    setup_logging()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map='auto'
    )
    # ).to('cuda')
    
    quant = QuipQuantizer(
        codebook=codebook,
        dataset="wikitext2",
        nsamples=1024, # 4096 - default ~500-750 CPU mem
        ft_train_size=256
    )

    quant.quantize_model(model, tokenizer, quant_dir)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--quant-dir", type=str, required=True)
    parser.add_argument("--codebook", type=str, required=True)
    parser.add_argument("--gpu-id", type=str, required=True)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpu_id}"
    Path(args.quant_dir).mkdir(parents=True, exist_ok=True)

    run_quantization(args.model_name, args.quant_dir, args.codebook)

if __name__ == "__main__":
    main()