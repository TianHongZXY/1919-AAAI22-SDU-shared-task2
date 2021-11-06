# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : run_finetune.sh
#   Last Modified : 2021-11-04 00:51
#   Describe      : 
#
# ====================================================
# CUDA_LAUNCH_BLOCKING=1
lang=$1
class=$2
pretrained_model=$3
gpu=$4
CUDA_VISIBLE_DEVICES=$4 python train.py \
    --gpus 1 \
    --max_epochs 20 \
    --lr 1e-5 \
    --train_batchsize 2 \
    --valid_batchsize 2 \
    --num_workers 8 \
    --val_check_interval 1.0 \
    --data_dir "data/$1/$2" \
    --model_name 'BertModel' \
    --pretrained_model "/home/zxy21/codes_and_data/.cache/pretrained_models/$3" \
    --warmup 0.1 \
    --pooler_type 'cls' \
    --accelerator 'ddp' \
    --finetune \
    --recreate_dataset
    # --child_tuning \
    # --checkpoint_path '/data/zxy/1919-AAAI22-SDU-shared-task2/save/BertModel/lr:1e-05-pooler_type:cls-pretrained_model:bert-base-cased-childtune:0-l2:0.0-finetune:0/epoch=01-valid_acc_epoch=0.4286.ckpt' \
    # --eval

