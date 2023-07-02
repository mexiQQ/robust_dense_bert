#!/bin/bash

target_folder=/hdd1/jianwei/workspace/robust_ticket_soups/dense/agnews/outputs2
for file in $target_folder/*
do
    echo $file
    dir_name=$(basename $file)

    CUDA_VISIBLE_DEVICES=0 python evaluate_robustness.py \
    --model_dir $file \
    --dataset_name ag_news \
    --epochs 30 \
    --result_file results/result_$dir_name.csv \
    --num_examples 200 \
    --seed 42
done
 
# target_folder=/hdd1/jianwei/workspace/robust_ticket_soups/dense/sparse/outputs2/finetune_glue-sst2_lr2e-05_epochs5_seed42_time1683247966961
# dir_name=$(basename $target_folder)
# python evaluate_robustness.py \
#     --model_dir $target_folder \
#     --dataset_name glue \
#     --task_name sst2 \
#     --epochs 5 \
#     --result_file sparse/results2/result_$dir_name.csv \
#     --num_examples 872 \
#     --seed 42
