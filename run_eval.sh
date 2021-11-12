# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : run_eval.sh
#   Last Modified : 2021-11-12 16:33
#   Describe      : 
#
# ====================================================

lang=$1
class=$2
gpu=$3
CUDA_VISIBLE_DEVICES=$3 python train.py \
    --gpus 1 \
    --max_epochs 20 \
    --lr 1e-5 \
    --train_batchsize 1 \
    --valid_batchsize 1 \
    --num_workers 8 \
    --val_check_interval 1.0 \
    --data_dir "data/$1/$2" \
    --model_name 'BertModel' \
    --warmup 0.1 \
    --pooler_type 'cls' \
    --accelerator 'ddp' \
    --pretrained_model "allenai/scibert_scivocab_uncased" \
    --checkpoint_path "/home/zxy21/codes_and_data/1919-AAAI22-SDU-shared-task2/save/BertModel/english/aux_scientific/bs=1-lr=1e-05-pooler_type=cls-pretrained_model=scibert_scivocab_uncased-childtune=1-l2=0.0-finetune=1-clip=1-dropout=0.5-adv=1-precision-16//epoch=02-valid_f1=0.6152.ckpt" \
    --eval \
    # --recreate_dataset
    # --checkpoint_path "/home/zxy21/codes_and_data/1919-AAAI22-SDU-shared-task2/save/BertModel/bs:2-lr:1e-05-pooler_type:cls-pretrained_model:roberta-large-childtune:0-l2:0.0-finetune:1-lang:english-class:scientific/epoch=01-valid_acc_epoch=0.3098.ckpt" \
