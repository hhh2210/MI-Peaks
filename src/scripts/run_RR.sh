#!/bin/bash
set -x 

cd applications/

PROMPT_TYPE="deepseek-math"
SPLIT="test"
NUM_TEST_SAMPLE=-1
LOG_DIR=log/$(date +%m-%d_%H-%M)
mkdir -p $LOG_DIR


gpu_counter=0


model_path='/data1/public/models/DeepSeek-R1-Distill-Llama-8B/'

datasets=(
    "aime24" 
)
ei_layers=(23)


model_name=$(basename "$model_path") 

for (( j=0; j<${#datasets[@]}; j++ )); do
    dataset=${datasets[$j]}
    
    for (( k=0; k<${#ei_layers[@]}; k++ )); do
        ei_layer=${ei_layers[$k]}
        save_dir="scores/${model_name}/${ei_layer}/"
        responses_dir="responses/${model_name}/${dataset}"
        mkdir -p "$save_dir"
        mkdir -p "$responses_dir"

        echo "Launched: model=$model_name, dataset=$dataset, layer=$ei_layer, GPU=$gpu_counter"

        log_file="${LOG_DIR}/${model_name}_${dataset}_${ei_layer}.log"
        mkdir -p "$(dirname "$log_file")"

        CUDA_VISIBLE_DEVICES=$gpu_counter \
        python RR_evaluate.py \
            --model_name_or_path "$model_path" \
            --data_names "$dataset" \
            --inject_layer_id $ei_layer \
            --extract_layer_id $ei_layer \
            --num_test_sample "$NUM_TEST_SAMPLE" \
            --output_file "$responses_dir/${ei_layer}.jsonl"\
            --use_recursive_thinking True \
            --num_recursive_steps 1 \
            --output_dir "$save_dir" \
            --split "$SPLIT" \
            --prompt_type "$PROMPT_TYPE" \
            --seed 0 \
            --temperature 0 \
            --n_sampling 1 \
            --top_p 1 \
            --start 0 \
            --end -1 \
            --save_outputs \
            --overwrite \
            --interested_tokens_file_path data/${model_name}.jsonl \
            --max_tokens_per_call 16000 >> "$log_file" 2>&1 &

        ((gpu_counter++))
    done
done

# Wait for all background tasks to complete
wait
echo "All tasks completed!"