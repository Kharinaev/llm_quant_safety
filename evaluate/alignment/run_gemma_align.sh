#!/bin/bash

CUDA_VISIBLE_DEVICES="1,3" python gemma_align.py \
    --datapath /root/llm_quant_safety/data/human_eval/mismatched_errors.csv \
    --savepath /root/llm_quant_safety/data/llm_eval/mismatched_errors_v3_2.csv \
    --prompt-template-file /root/llm_quant_safety/evaluate/alignment/prompts/prompt_template_v3.md \
    --max-tokens 512