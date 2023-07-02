#!/bin/bash

# target_folder=/hdd1/jianwei/workspace/robust_ticket_soups/dense/sparse/outputs2
# for file in $target_folder/*
# do
#     echo $file
#     dir_name=$(basename $file)

#     python evaluate_robustness.py \
#     --model_dir $file \
#     --dataset_name glue \
#     --task_name sst2 \
#     --epochs 5 \
#     --result_file sparse/results2/result_$dir_name.csv \
#     --num_examples 872 \
#     --seed 42
# done
 
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

textattack attack --model /hdd1/jianwei/workspace/robust_ticket_soups/dense/sparse/outputs3/finetune_glue-sst2_lr2e-07_epochs5_seed42_time1683264869343/epoch0 --recipe textfooler --dataset-from-huggingface sst2 --dataset-split validation --num-examples -1 --random-seed 42 --parallel
