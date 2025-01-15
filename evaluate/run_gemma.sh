#!/bin/bash

CUDA_VISIBLE_DEVICES="2,3" python run_gemma.py \
    --datapath /root/llm_quant_safety/data/abliterated_en_responses.csv \
    --savepath /root/llm_quant_safety/data/llm_eval/gemma_eval_abliterated_en_responses.csv \
    --prompt-template-file /root/llm_quant_safety/evaluate/alignment/prompts/prompt_template_v3.md \
    --max-tokens 512