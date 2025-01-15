#!/bin/bash

total_examples=3738
processed_examples=0
start_time=$(date +%s)
log_file="output_2.log"

log_and_print() {
    echo "$1"
    echo "$1" >> "$log_file"
}

format_time() {
    local seconds=$1
    printf "%02d:%02d:%02d" $((seconds/3600)) $((seconds%3600/60)) $((seconds%60))
}

cp logs/output_2.log logs/output_prev_2.log
> logs/output_2.log

while [ $processed_examples -lt $total_examples ]
do
    # Запускаем Python-скрипт и сохраняем вывод
    output=$(python3 run_model_quip.py --model-name "/root/llm_quant_safety/quantization/models/Llama-3.1-8B-Instruct-quip-2bit" --datapath "~/llm_quant_safety/data/parts/p2.csv" --bs 64 --gpu-id 3 | tee /dev/tty)
    
    # Получаем количество обработанных примеров из вывода
    examples_this_run=$(echo "$output" | tail -n 1)
    
    # Обновляем общее количество обработанных примеров
    processed_examples=$((processed_examples + examples_this_run))
    
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    elapsed_formatted=$(format_time $elapsed_time)
    
    # Расчет оставшегося времени
    remaining_examples=$((total_examples - processed_examples))
    time_per_example=$(awk "BEGIN {print $elapsed_time / $processed_examples}")
    estimated_remaining_time=$(awk "BEGIN {print int($time_per_example * $remaining_examples)}")
    remaining_formatted=$(format_time $estimated_remaining_time)
    
    log_and_print "Processed $processed_examples out of $total_examples examples"
    log_and_print "Elapsed time: $elapsed_formatted"
    log_and_print "Estimated remaining time: $remaining_formatted"
    log_and_print "----------------------------------------"
    
    # Небольшая пауза перед следующим запуском
    sleep 5
done

total_time=$(($(date +%s) - start_time))
total_formatted=$(format_time $total_time)

log_and_print "All examples processed!"
log_and_print "Total time taken: $total_formatted"