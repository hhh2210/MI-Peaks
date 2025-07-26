#!/bin/bash

# Activate the mi environment
source /dockerdata/llm_eval/mi/bin/activate

export TOKENIZERS_PARALLELISM=false

model="/data1/public/models/DeepSeek-R1-Distill-Llama-8B/"

model_name=$(basename "$model")
dataset="aime24"

gpu_id=4,5,6,7

# math  512 772 1024 2048 3072 4096 5120
# aime 1024 2048 3072 4096 6144 8192 12288
token_budgets=(1024 2048 3072 4096 6144 8192 12288)

# Repeat generation parameters
repeat_prompt="Let me reconsider the original question."
continuation_prompt="Now I need to"

# Create results summary file in applications directory
cd applications/
summary_file="results_summary_repeat_${dataset}_$(date +%Y%m%d_%H%M%S).txt"
echo "=== Repeat Evaluation Results Summary ===" > "$summary_file"
echo "Model: $model_name" >> "$summary_file"
echo "Dataset: $dataset" >> "$summary_file"
echo "Time: $(date)" >> "$summary_file"
echo "=========================================" >> "$summary_file"

# Store results for final summary
declare -A results_acc
declare -A results_details

# Loop through each token budget
for token_budget in "${token_budgets[@]}"; do
    echo "Running Repeat with token_budget: $token_budget"
    
    save_dir="results/${model_name}/${dataset}_repeat_budget${token_budget}/"
    mkdir -p "$save_dir"
    
    # Run the evaluation and capture output
    output=$(CUDA_VISIBLE_DEVICES=$gpu_id python3 -u repeat.py \
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
        --max_tokens_per_call 4096 \
        --token_budget $token_budget \
        --repeat_prompt "$repeat_prompt" \
        --continuation_prompt "$continuation_prompt" 2>&1)
    
    # Extract accuracy from output
    acc=$(echo "$output" | grep -oP "acc':\s*np\.float64\(\K[0-9.]+(?=\))" | tail -1)
    if [ -z "$acc" ]; then
        acc=$(echo "$output" | grep -oP "acc':\s*\K[0-9.]+" | tail -1)
    fi
    
    # Extract full result dict
    result_dict=$(echo "$output" | grep -oP "\{'num_samples':.*?\}" | tail -1)
    
    # Store results
    results_acc[$token_budget]=$acc
    results_details[$token_budget]=$result_dict
    
    echo "Completed token_budget: $token_budget (Accuracy: $acc%)"
    echo "--------------------------------"
    
    # Save to summary file
    echo "" >> "$summary_file"
    echo "Token Budget: $token_budget" >> "$summary_file"
    echo "Result: $result_dict" >> "$summary_file"
    echo "Accuracy: $acc%" >> "$summary_file"
done

# Print final summary
echo ""
echo "============================================"
echo "         FINAL RESULTS SUMMARY              "
echo "============================================"
echo "Model: $model_name"
echo "Dataset: $dataset"
echo ""
printf "%-15s %-10s\n" "Token Budget" "Accuracy"
printf "%-15s %-10s\n" "------------" "--------"

# Also append to summary file
echo "" >> "$summary_file"
echo "============================================" >> "$summary_file"
echo "         FINAL RESULTS SUMMARY              " >> "$summary_file"
echo "============================================" >> "$summary_file"
printf "%-15s %-10s\n" "Token Budget" "Accuracy" >> "$summary_file"
printf "%-15s %-10s\n" "------------" "--------" >> "$summary_file"

for token_budget in "${token_budgets[@]}"; do
    acc=${results_acc[$token_budget]}
    if [ -z "$acc" ]; then
        acc="N/A"
    fi
    printf "%-15s %-10s\n" "$token_budget" "${acc}%"
    printf "%-15s %-10s\n" "$token_budget" "${acc}%" >> "$summary_file"
done

echo ""
echo "Detailed results saved to: $summary_file"
echo "All token budgets completed!"