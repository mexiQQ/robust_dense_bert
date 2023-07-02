#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

log_file='./logs2/seed-107-lr-3e-5-epoch-30-imdb-freelb-norm_for_each_word_no-dropout'  # training log
# result_file='seed-42-lr-2e-5-epoch-30-sst2.csv'  # attack results
cpkt='./outputs2/' # Model saving path
seed=426
lr=3e-5
epoch=30
model='/hdd1/jianwei/workspace/robust_ticket_soups/vendor/bert-base-uncased'

CUDA_VISIBLE_DEVICES=2 python run_glue_freelb_norm_for_each_word.py \
    --model_name $model \
    --ckpt_dir $cpkt \
    --dataset_name imdb \
    --epochs $epoch \
    --num_labels 2 \
    --adv_lr 0.03 \
    --adv_steps 3 \
    --bsz 32 \
    --num_examples 100 \
    --warmup_ratio 0.002 \
    --lr $lr \
    --seed $seed \
    --max_seq_length 256 \
    #--force_overwrite 1 >> ${log_file}
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