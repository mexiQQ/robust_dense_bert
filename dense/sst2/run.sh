#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

log_file='./logs/seed-42-lr-2e-5-epoch-5-sst2-from-robust.log'  # training log
result_file='seed-42-lr-2e-5-epoch-5-sst2-from-robust.csv'  # attack results
cpkt='./outputs/' # Model saving path
seed=42
lr=2e-5
epoch=5
model='/hdd1/jianwei/workspace/robust_ticket/save_models/fine-tune/finetune_bert-base-uncased_glue-sst2_lr2e-05_epochs25_seed42/epoch24'

# SST-2
python run_glue.py \
--model_name $model \
--ckpt_dir $cpkt \
--dataset_name glue \
--task_name sst2 \
--epochs $epoch \
--result_file ${result_file} \
--num_examples 872 \
--lr $lr \
--seed $seed \
--max_seq_length 128 \
--force_overwrite 1 >> ${log_file}

log_file='./logs/seed-42-lr-2e-5-epoch-10-sst2-from-robust.log'  # training log
result_file='seed-42-lr-2e-5-epoch-10-sst2-from-robust.csv'  # attack results
cpkt='./outputs/' # Model saving path
seed=42
lr=2e-5
epoch=10
model='/hdd1/jianwei/workspace/robust_ticket/save_models/fine-tune/finetune_bert-base-uncased_glue-sst2_lr2e-05_epochs25_seed42/epoch24'

python run_glue.py \
--model_name $model \
--ckpt_dir $cpkt \
--dataset_name glue \
--task_name sst2 \
--epochs $epoch \
--result_file ${result_file} \
--num_examples 872 \
--lr $lr \
--seed $seed \
--max_seq_length 128 \
--force_overwrite 1 >> ${log_file}

log_file='./logs/seed-42-lr-3e-5-epoch-5-sst2-from-robust.log'  # training log
result_file='seed-42-lr-3e-5-epoch-5-sst2-from-robust.csv'  # attack results
cpkt='./outputs/' # Model saving path
seed=42
lr=3e-5
epoch=5
model='/hdd1/jianwei/workspace/robust_ticket/save_models/fine-tune/finetune_bert-base-uncased_glue-sst2_lr2e-05_epochs25_seed42/epoch24'

python run_glue.py \
--model_name $model \
--ckpt_dir $cpkt \
--dataset_name glue \
--task_name sst2 \
--epochs $epoch \
--result_file ${result_file} \
--num_examples 872 \
--lr $lr \
--seed $seed \
--max_seq_length 128 \
--force_overwrite 1 >> ${log_file}

log_file='./logs/seed-426-lr-2e-5-epoch-10-sst2-from-robust.log'  # training log
result_file='seed-426-lr-2e-5-epoch-10-sst2-from-robust.csv'  # attack results
cpkt='./outputs/' # Model saving path
seed=426
lr=2e-5
epoch=10
model='/hdd1/jianwei/workspace/robust_ticket/save_models/fine-tune/finetune_bert-base-uncased_glue-sst2_lr2e-05_epochs25_seed42/epoch24'

python run_glue.py \
--model_name $model \
--ckpt_dir $cpkt \
--dataset_name glue \
--task_name sst2 \
--epochs $epoch \
--result_file ${result_file} \
--num_examples 872 \
--lr $lr \
--seed $seed \
--max_seq_length 128 \
--force_overwrite 1 >> ${log_file}

# # IMDB
# python run_glue.py \
# --model_name $model \
# --ckpt_dir $cpkt \
# --dataset_name imdb \
# --num_labels 2 \
# --bsz 32 \
# --epochs $epoch \
# --lr $lr \
# --seed $seed \
# --max_seq_length 256 \
# --result_file $result_file \
# --num_examples 100 \
# --force_overwrite 1 >> ${log_file}


# # AGNEWS
# python run_glue.py \
# --model_name $model \
# --ckpt_dir $cpkt \
# --dataset_name ag_news \
# --num_labels 4 \
# --bsz 32 \
# --epochs $epoch \
# --lr $lr \
# --seed $seed \
# --max_seq_length 256 \
# --result_file $result_file \
# --num_examples 200 \
# --force_overwrite 1 >> ${log_file}