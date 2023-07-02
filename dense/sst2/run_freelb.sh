#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

log_file='./logs7/seed-426-lr-2e-5-epoch-30-sst2-freelb-adv-3e-2_norm_for_each_word.log-ads-5-no-dropout'  # training log
# result_file='seed-42-lr-2e-5-epoch-30-sst2.csv'  # attack results
cpkt='./outputs7/' # Model saving path
seed=426
lr=2e-5
epoch=40
model='/hdd1/jianwei/workspace/robust_ticket_soups/vendor/bert-base-uncased'

CUDA_VISIBLE_DEVICES=2 python run_glue_freelb_norm_for_each_word.py \
    --model_name $model \
    --ckpt_dir $cpkt \
    --dataset_name glue \
    --task_name sst2 \
    --epochs $epoch \
    --num_examples 872 \
    --lr $lr \
    --bsz 32 \
    --adv_lr 0.03 \
    --adv_steps 5 \
    --seed $seed \
    --max_seq_length 128 \
    --force_overwrite 1 >> ${log_file}
# --result_file ${result_file} \


# log_file='./logs4/seed-42-lr-2e-5-epoch-30-sst2-freelb.log'  # training log
# # result_file='seed-42-lr-2e-5-epoch-30-sst2.csv'  # attack results
# cpkt='./outputs4/' # Model saving path
# seed=42
# lr=2e-5
# epoch=30
# model='bert-base-uncased'

# CUDA_VISIBLE_DEVICES=1 python run_glue_freelb.py \
#     --model_name $model \
#     --ckpt_dir $cpkt \
#     --dataset_name glue \
#     --task_name sst2 \
#     --epochs $epoch \
#     --num_examples 872 \
#     --lr $lr \
#     --seed $seed \
#     --max_seq_length 128 \
#     --force_overwrite 1 >> ${log_file}