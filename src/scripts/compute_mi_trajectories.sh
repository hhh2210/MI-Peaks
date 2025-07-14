


models=(
    deepseek-ai/DeepSeek-R1-Distill-Llama-8B
)

for model in "${models[@]}"; do
    # 1. generate representations
    echo generate reps on $model ...
    python generate_activation.py --model $model --layers 31 --dataset math_train_12k --sample_num 100
    python generate_gt_activation.py --model $model --layers 31 --dataset math_train_12k --sample_num 100

    # 2. compute mutual information
    echo compute mi on $model ...
    python cal_mi.py --gt_model $model --test_model $model --layers 31 --dataset math_train_12k  --sample_num 100 &

done

