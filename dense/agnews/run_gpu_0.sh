log_file='./logs/seed-302-lr-2e-5-epoch-30-agnews.log'  # training log
# result_file='seed-42-lr-2e-5-epoch-30-sst2.csv'  # attack results
cpkt='./outputs/' # Model saving path
seed=302
lr=2e-5
epoch=30
model='bert-base-uncased'

CUDA_VISIBLE_DEVICES=0 python run_glue.py \
    --model_name $model \
    --ckpt_dir $cpkt \
    --dataset_name ag_news \
    --epochs $epoch \
    --num_labels 4 \
    --bsz 32 \
    --num_examples 200 \
    --lr $lr \
    --seed $seed \
    --max_seq_length 256 \
    --force_overwrite 1 >> ${log_file}
# --result_file ${result_file} \

# log_file='./logs3/seed-0-lr-2e-5-epoch-30-sst2-expr1.log'  # training log
# # result_file='seed-42-lr-2e-5-epoch-30-sst2.csv'  # attack results
# cpkt='./outputs3/' # Model saving path
# seed=0
# lr=2e-5
# epoch=30
# model='bert-base-uncased'

# CUDA_VISIBLE_DEVICES=0 python run_glue.py \
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

# log_file='./logs3/seed-0-lr-2e-5-epoch-30-sst2-expr2.log'  # training log
# # result_file='seed-42-lr-2e-5-epoch-30-sst2.csv'  # attack results
# cpkt='./outputs3/' # Model saving path
# seed=0
# lr=2e-5
# epoch=30
# model='bert-base-uncased'

# CUDA_VISIBLE_DEVICES=0 python run_glue.py \
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

# log_file='./logs3/seed-0-lr-2e-5-epoch-30-sst2-expr3.log'  # training log
# # result_file='seed-42-lr-2e-5-epoch-30-sst2.csv'  # attack results
# cpkt='./outputs3/' # Model saving path
# seed=0
# lr=2e-5
# epoch=30
# model='bert-base-uncased'

# CUDA_VISIBLE_DEVICES=0 python run_glue.py \
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

# log_file='./logs3/seed-0-lr-2e-5-epoch-30-sst2-expr4.log'  # training log
# # result_file='seed-42-lr-2e-5-epoch-30-sst2.csv'  # attack results
# cpkt='./outputs3/' # Model saving path
# seed=0
# lr=2e-5
# epoch=30
# model='bert-base-uncased'

# CUDA_VISIBLE_DEVICES=0 python run_glue.py \
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
