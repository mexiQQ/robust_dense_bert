!/bin/bash

target_folder=/hdd1/jianwei/workspace/robust_ticket_soups/dense/imdb/outputs2
for file in $target_folder/*
do
    echo $file
    dir_name=$(basename $file)

    CUDA_VISIBLE_DEVICES=3 python evaluate_robustness.py \
    --model_dir $file \
    --dataset_name imdb \
    --epochs 30 \
    --result_file results2/result_$dir_name.csv \
    --num_examples 100 \
    --seed 42
done
 
# target_folder=/hdd1/jianwei/workspace/robust_ticket_soups/dense/imdb/outputs/finetune_imdb_lr2e-05_epochs30_seed426_time1687042137909
# dir_name=$(basename $target_folder)
# CUDA_VISIBLE_DEVICES=3 python evaluate_robustness.py \
#     --model_dir $target_folder \
#     --dataset_name imdb \
#     --epochs 30 \
#     --result_file results/result_$dir_name.csv \
#     --num_examples 100 \
    # --seed 42
