
# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : run.sh
#   Last Modified : 2021-10-15 13:50
#   Describe      : 
#
# ====================================================

CUDA_VISIBLE_DEVICES=0,1,2 python train.py \
    --gpus 3 \
    --max_epochs 100 \
    --lr 1e-5 \
    --train_batchsize 16 \
    --valid_batchsize 16 \
    --num_workers 8 \
    --val_check_interval 1.0 \
    --data_dir './data/english/legal' \
    --pretrained_model 'princeton-nlp/sup-simcse-roberta-base' \
    --model_name 'BertModel' \
    --warmup 0.1 \
    --pooler_type 'avg_top2' \
    --accelerator 'ddp' \








