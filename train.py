# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : train.py
#   Last Modified : 2021-10-17 09:30
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


def main():
    device = torch.device(args.device) if torch.cuda.is_available() and args.device != -1 else torch.device('cpu')

    model = AutoModel.from_pretrained(args.model)
    args.hidden_size = model.config.hidden_size
    print(model.config)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_sent, train_abbr, train_long_term, train_label = read_data(args.trainset)
    valid_sent, valid_abbr, valid_long_term, valid_label = read_data(args.validset)

    tokenizer.add_tokens(train_abbr)
    model.resize_token_embeddings(len(tokenizer))

    #  init_model(model, args)
    model = model.to(device)

    train_encoded_inputs = tokenizer(train_sent, train_long_term, padding=True, truncation=True, return_tensors="pt")
    valid_encoded_inputs = tokenizer(valid_sent, valid_long_term, padding=True, truncation=True, return_tensors="pt")
    for k, v in train_encoded_inputs.items(): 
        train_encoded_inputs[k] = train_encoded_inputs[k].to(args.device)
    for k, v in valid_encoded_inputs.items(): 
        valid_encoded_inputs[k] = valid_encoded_inputs[k].to(args.device)

    train_dataset = CrossInteractionDataset(train_encoded_inputs, torch.FloatTensor(train_label))
    valid_dataset = CrossInteractionDataset(valid_encoded_inputs, torch.FloatTensor(valid_label))

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              )
    valid_loader = DataLoader(valid_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            )

    #  optimizer_warmup = AdamW(model.parameters(), lr=args.warm_up_learning_rate)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    step = 0
    start_epoch = int(args.checkpoint.split("_")[-1]) if args.load_checkpoint else 0

    for epoch in range(start_epoch, args.total_epochs):
        print('\nTRAINING EPOCH %d' % epoch)

        for idx, batch in enumerate(train_loader):
            step += 1
            #  optimizer_warmup.zero_grad()
            optimizer.zero_grad()

            outputs = model(**batch['data'])
            pooler_output = model._pooler(batch['data']['attention_mask'], outputs)
            # If using "cls", we add an extra MLP layer
            # (same as BERT's original implementation) over the representation.
            if model.pooler_type == "cls":
                pooler_output = model.mlp(pooler_output)

            logits = model.output(pooler_output)
            loss = criterion(logits.view(-1), batch['label'].to(device))


            loss_prt = loss.cpu().detach().numpy() if CUDA_AVAILABLE else loss.detach().numpy()
            loss_prt = round(float(loss_prt), 3)

            loss.backward()
            optimizer.step()
            if step % args.print_frequency == 0 and not args.print_frequency == -1:
                print(f"Train step {step}\tLoss: {loss_prt}")

            if step % (args.print_frequency * 10) == 0 and not args.print_frequency == -1:
                print('Evaluating...')
                model.eval()
                for idx, batch in enumerate(valid_loader):
                    outputs = model(**batch['data'])
                    pooler_output = model._pooler(batch['data']['attention_mask'], outputs)
                    if model.pooler_type == "cls":
                        pooler_output = model.mlp(pooler_output)

                    logits = model.output(pooler_output)
                    loss = criterion(logits.view(-1), batch['label'].to(device))
                    loss_prt = loss.cpu().detach().numpy() if CUDA_AVAILABLE else loss.detach().numpy()
                    loss_prt = round(float(loss_prt), 3)
                    print(f"Valid Loss: {loss_prt}")
                model.train()

        print(f'Saving model at epoch {epoch} step {step}')
        model.save_pretrained(f"{args.save_model_path}_%d" % epoch)

            #  if step <= args.warm_up_steps:
            #      if step % 500 == 0:
            #          print(f"warm up step {step}\tLoss: {loss_prt}")
            #      loss.backward()
            #      optim_warmup.step()
            #  else:
            #      if step % 500 == 0:
            #          print(f"train step {step}\tL_nll_d1: {loss_prt}, L_nll_d2: {loss_2_prt} and L_ul: {ul_loss_prt}")
            #      (loss + 0.01 * loss_2 + 0.01 * ul_loss).backward()
            #      optim.step()


def main1(args):

    save_path = os.path.join(args.save_dir, args.model_name)

    if args.model_name == 'BertModel':
        Model = Bert
        hyparas = 'lr: {} - pooler_type: {} - pretrained_model: {}'.format(
            args.lr, args.pooler_type, os.path.split(args.pretrained_model)[-1])
        save_path = os.path.join(save_path, hyparas)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    seed_everything(args.seed)

    logger = loggers.TensorBoardLogger(save_dir=os.path.join(
        save_path, 'logs/'), name='')
    checkpoint = ModelCheckpoint(dirpath=save_path,
                                 save_top_k=1,
                                 monitor='valid_acc',
                                 mode='min',
                                 filename='{epoch:02d}-{valid_acc:.4f}')
    early_stop = EarlyStopping(monitor='valid_acc', mode='max', patience=5)
    trainer = Trainer.from_argparse_args(args,
                                         logger=logger,
                                         callbacks=[checkpoint, early_stop])

    if args.eval is False:
        print(args.pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model,
                                                  use_fast=True)

        data_model = Task2DataModel(args, tokenizer)
        args.nlabels = data_model.max_long_form
        model = Model(args, tokenizer)
        trainer.fit(model, data_model)
        tokenizer.save_pretrained(save_path)
        checkpoint_path = checkpoint.best_model_path
    else:
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        data_model = Task2DataModel(args, tokenizer)
        checkpoint_path = os.path.join(save_path, args.checkpoint_path)

    # module evaluation
    model = Model.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)
    evaluation(args, model, data_model, save_path)

def evaluation(args, model, data_model, save_path):

    data_model.setup('test')
    tokenizer = data_model.tokenizer
    test_loader = data_model.test_dataloader()

    device = torch.device('cpu')
    model.to(device)
    model.eval()

    results = []
    for batch in tqdm(test_loader):

        predicts = model.predict(batch['input_ids'].to(device),
                                 batch['attention_mask'].to(device),
                                 batch['token_type_ids'].to(device),
                                 batch['softmax_mask'].to(device))

        for idx, predict in enumerate(predicts):
            long_form = data_model.acronym2lf[batch['acronym'][idx]][int(predict)]

            pred = {
                'ID': batch['idx'][idx],
                'label': long_form
            }
            results.append(pred)

    with open(os.path.join(save_path, '-'.join(args.data_dir.split('/')[2:]) + '-outputs.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    total_parser = argparse.ArgumentParser("AAAI task2 AD")

    # * Args for data preprocessing
    total_parser = Task2DataModel.add_data_specific_args(total_parser)
    # * Args for training
    total_parser = Trainer.add_argparse_args(total_parser)
    # * Args for model specific
    total_parser = BaseADModel.add_model_specific_args(total_parser)

    args = total_parser.parse_args()
    
    main1(args)
