
# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : run.sh
#   Last Modified : 2021-10-17 08:04
#   Describe      : 
#
# ====================================================

# CUDA_VISIBLE_DEVICES=0,1,2 CUDA_LAUNCH_BLOCKING=1
python train.py \
    --gpus 1 \
    --max_epochs 100 \
    --lr 1e-5 \
    --train_batchsize 32 \
    --valid_batchsize 16 \
    --num_workers 8 \
    --val_check_interval 1.0 \
    --data_dir './data/english/legal' \
    --model_name 'BertModel' \
    --pretrained_model 'bert-base-cased' \
    --warmup 0.1 \
    --pooler_type 'cls' \
    --accelerator 'ddp' \
    # --eval
    # --recreate_dataset








