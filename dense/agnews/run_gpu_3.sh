log_file='./logs2/seed-42-lr-2e-5-epoch-30-sst2-exper-2.log'  # training log
# result_file='seed-42-lr-2e-5-epoch-30-sst2.csv'  # attack results
cpkt='./outputs2/' # Model saving path
seed=42
lr=2e-5
epoch=30
model='bert-base-uncased'

CUDA_VISIBLE_DEVICES=3 python run_glue.py \
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

# log_file='./logs3/seed-0-lr-2e-5-epoch-30-sst2-expr13.log'  # training log
# # result_file='seed-42-lr-2e-5-epoch-30-sst2.csv'  # attack results
# cpkt='./outputs3/' # Model saving path
# seed=0
# lr=2e-5
# epoch=30
# model='bert-base-uncased'

# CUDA_VISIBLE_DEVICES=3 python run_glue.py \
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

# log_file='./logs3/seed-0-lr-2e-5-epoch-30-sst2-expr14.log'  # training log
# # result_file='seed-42-lr-2e-5-epoch-30-sst2.csv'  # attack results
# cpkt='./outputs3/' # Model saving path
# seed=0
# lr=2e-5
# epoch=30
# model='bert-base-uncased'

# CUDA_VISIBLE_DEVICES=3 python run_glue.py \
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

# log_file='./logs3/seed-0-lr-2e-5-epoch-30-sst2-expr15.log'  # training log
# # result_file='seed-42-lr-2e-5-epoch-30-sst2.csv'  # attack results
# cpkt='./outputs3/' # Model saving path
# seed=0
# lr=2e-5
# epoch=30
# model='bert-base-uncased'

# CUDA_VISIBLE_DEVICES=3 python run_glue.py \
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

# log_file='./logs3/seed-0-lr-2e-5-epoch-30-sst2-expr16.log'  # training log
# # result_file='seed-42-lr-2e-5-epoch-30-sst2.csv'  # attack results
# cpkt='./outputs3/' # Model saving path
# seed=0
# lr=2e-5
# epoch=30
# model='bert-base-uncased'

# CUDA_VISIBLE_DEVICES=3 python run_glue.py \
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