#!/bin/bash

cd applications/

model="your_dir/DeepSeek-R1-Distill-Llama-8B"

model_name=$(basename "$model")
dataset=aime24

gpu_id=0

token_budget=4096

save_dir="results/${model_name}/${dataset}_budget${token_budget}/"
mkdir -p "$save_dir"


CUDA_VISIBLE_DEVICES=$gpu_id python3 -u TTTS_evaluate.py \
    --model_name_or_path $model \
    --data_names $dataset \
    --output_dir $save_dir \
    --split "test" \
    --prompt_type "deepseek-math" \
    --num_test_sample -1 \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --overwrite \
    --use_vllm \
    --thinking_tokens_file_path data/${model_name}.jsonl \
    --max_tokens_per_call 4096 \
    --token_budget $token_budget
