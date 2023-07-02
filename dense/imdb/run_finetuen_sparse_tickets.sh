#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

log_file='./sparse/logs3/seed-42-lr-2e-7-epoch-5-sst2-from-robust-2x.log'  # training log
# result_file='seed-42-lr-2e-5-epoch-5-sst2-from-robust.csv'  # attack results
cpkt='./sparse/outputs3' # Model saving path
seed=42
lr=2e-7
epoch=5
model='/hdd1/jianwei/workspace/OBC/bert/outputs_2x_3'

# SST-2
python finetune_sparse_tickets.py \
--model_name $model \
--ckpt_dir $cpkt \
--dataset_name glue \
--task_name sst2 \
--epochs $epoch \
--num_examples 872 \
--lr $lr \
--seed $seed \
--max_seq_length 128 \
--force_overwrite 1 >> ${log_file}
# --result_file ${result_file} \