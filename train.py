# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : train.py
#   Last Modified : 2021-11-07 17:38
#   Describe      : 
#
# ====================================================

import torch
import json
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import AutoModel, AutoTokenizer, AdamW
from dataloader import Task2DataModel, Task2Dataset
from itertools import chain
from models.bert_baseline import Bert, BaseADModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything, loggers
from pytorch_lightning.callbacks.progress import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def main(args):

    save_path = os.path.join(args.save_dir, args.model_name)

    if args.checkpoint_path is not None:
        save_path = os.path.split(args.checkpoint_path)[0]
    else:
        hyparas = '{}/{}/bs={}-lr={}-pooler_type={}-pretrained_model={}-childtune={}-l2={}-finetune={}-clip={}-dropout={}-adv={}-precision-{}'.format(
            args.data_dir.split('/')[1], args.data_dir.split('/')[2], args.train_batchsize, args.lr, args.pooler_type, 
            os.path.split(args.pretrained_model)[-1], int(args.child_tuning), args.l2, int(args.finetune), args.gradient_clip_val, args.mlp_dropout, int(args.adv), 
            args.precision)
        save_path = os.path.join(save_path, hyparas)

    args.pretrained_model_name = args.pretrained_model
    args.pretrained_model = os.path.join("/home/zxy21/codes_and_data/.cache/pretrained_models/", args.pretrained_model)
    Model = Bert
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    seed_everything(args.seed)

    checkpoint = ModelCheckpoint(dirpath=save_path,
                                 save_top_k=3,
                                 monitor='valid_f1',
                                 mode='max',
                                 filename='{epoch:02d}-{valid_f1:.4f}')
    early_stop = EarlyStopping(monitor='valid_f1', mode='max', patience=5)
    logger = loggers.TensorBoardLogger(save_dir=os.path.join(save_path, 'logs/'), name='')
    trainer = Trainer.from_argparse_args(args,
                                         logger=logger,
                                         callbacks=[checkpoint, early_stop])

    print('-' * 30 + 'Args' + '-' * 30)
    for k, v in vars(args).items():
        print(k, ":", v, end=',\t')
    print('\n' + '-' * 64)
    if args.eval:
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        data_model = Task2DataModel(args, tokenizer)
        checkpoint_path = args.checkpoint_path
    else:
        print(args.pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model,
                                                  use_fast=True)
        data_model = Task2DataModel(args, tokenizer)
        args.nlabels = data_model.max_long_form
        model = Model(args, tokenizer)
        trainer.fit(model, data_model)
        tokenizer.save_pretrained(save_path)
        checkpoint_path = checkpoint.best_model_path

    # module evaluation
    print("Load checkpoint from {}".format(checkpoint_path))
    model = Model.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)

    output_save_path = os.path.join(save_path, 'dev-' + '-'.join(args.data_dir.split('/')[2:]) + '-outputs.json')
    evaluation(args, model, data_model, output_save_path, mode='fit')
    from scorer import run_evaluation
    p, r, f1 = run_evaluation(gold=os.path.join(args.data_dir, 'dev.json'), pred=output_save_path)
    print('Official Scores:')
    print('P: {:.2%}, R: {:.2%}, F1: {:.2%}'.format(p,r,f1))
    output_save_path = os.path.join(save_path, 'dev_F1_{:.3f}_test_output.json'.format(f1))
    evaluation(args, model, data_model, output_save_path, mode='test')

def evaluation(args, model, data_model, save_path, mode):
    data_model.setup(mode)
    tokenizer = data_model.tokenizer
    if mode == 'fit':
        test_loader = data_model.val_dataloader()
    else:
        test_loader = data_model.test_dataloader()

    #  device = torch.device('cuda:7')
    #  model.to(device)
    model.cuda()
    model.eval()

    results = []
    for batch in tqdm(test_loader):

        predicts = model.predict(batch['input_ids'].cuda(), #to(device),
                                 batch['attention_mask'].cuda(),# .to(device),
                                 batch['token_type_ids'].cuda(), #to(device),
                                 batch['softmax_mask'].cuda()) #to(device))

        for idx, predict in enumerate(predicts):
            long_form = data_model.acronym2lf_padded[batch['acronym'][idx]][int(predict)]

            pred = {
                'ID': batch['idx'][idx],
                'label': long_form
            }
            results.append(pred)

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print("Evaluation file saved at {}".format(save_path))


if __name__ == '__main__':
    total_parser = argparse.ArgumentParser("AAAI task2 AD")

    # * Args for data preprocessing
    total_parser = Task2DataModel.add_data_specific_args(total_parser)
    # * Args for training
    total_parser = Trainer.add_argparse_args(total_parser)
    # * Args for model specific
    total_parser = BaseADModel.add_model_specific_args(total_parser)

    args = total_parser.parse_args()
    
    main(args)

