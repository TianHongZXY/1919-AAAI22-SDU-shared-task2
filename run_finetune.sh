# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : run_finetune.sh
#   Last Modified : 2021-11-12 16:33
#   Describe      : 
#
# ====================================================
# CUDA_LAUNCH_BLOCKING=1
lang=$1
class=$2
pretrained_model=$3
CUDA_VISIBLE_DEVICES=$4 
export CUDA_VISIBLE_DEVICES

for bert_lr in 5e-6 1e-5 3e-5 5e-5
do
    for lr in 1e-4 3e-4 5e-4
    do
        CUDA_VISIBLE_DEVICES=$4 python train.py \
            --gpus 1\
            --max_epochs 100 \
            --lr ${lr} \
            --bert_lr ${bert_lr} \
            --train_batchsize 1 \
            --valid_batchsize 1 \
            --num_workers 8 \
            --data_dir "data/$1/$2" \
            --model_name 'BertModel' \
            --pretrained_model $3 \
            --warmup 0.1 \
            --pooler_type 'cls' \
            --finetune \
            --accelerator 'ddp' \
            --gradient_clip_val 1 \
            --precision 32 \
            --val_check_interval 1.0 \
            --adv \
            --mlp_dropout 0. 
    done
done
            # --checkpoint_path '/mnt/data_16TB/zxy21/1919-AAAI22-SDU-shared-task2/save/BertModel/english/aux_scientific/bs=1-lr=1e-05-pooler_type=cls-pretrained_model=scibert_scivocab_uncased-childtune=1-l2=0.0-finetune=1-clip=1-dropout=0.5-adv=1-precision-16/epoch=02-valid_f1=0.6172.ckpt'
            # --eval
            # --track_grad_norm 2 \
            # --child_tuning \
            # --recreate_dataset
            # --plugins "fsdp" \
