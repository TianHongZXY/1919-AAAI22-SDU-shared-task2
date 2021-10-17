# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : base_model.py
#   Last Modified : 2021-10-17 08:04
#   Describe      : 
#
# ====================================================


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class OutputLayer(nn.Module):

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, features, **kwargs):
        logits = self.dense(features)

        return logits


class BaseADModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseADModel')
        
        # * Args for general setting
        parser.add_argument("--pooler_type", type=str, default="cls")
        parser.add_argument('--eval', action='store_true', default=False)
        parser.add_argument('--checkpoint_path', default=None, type=str)
        parser.add_argument('--seed', default=20020206, type=int)
        parser.add_argument('--save_dir', default='./save', type=str)
        parser.add_argument('--model_name', default='BaseADModel', type=str)
        parser.add_argument('--pretrained_model',
                            default='bert-base-cased',
                            type=str)
        
        parser.add_argument('--lr', default=1e-5, type=float)
        parser.add_argument('--warmup', default=0.1, type=float)


        return parent_args

    def __init__(self, args, tokenizer):
        super().__init__()

        if isinstance(args, dict):
            args = argparse.Namespace(**args)

        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.hidden_size = self.config.hidden_size
        self.nlabels = args.nlabels 
        self.loss_fn = nn.CrossEntropyLoss()

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.train_dataloader()
            self.total_step = int(self.trainer.max_epochs * len(train_loader) / \
                (max(1, self.trainer.gpus) * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)

    def train_inputs(self, batch):
        batch_data = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'softmax_mask': batch['softmax_mask'],
            'token_type_ids': batch['token_type_ids']
        }
        return batch_data

    def training_step(self, batch, batch_idx):
        inputs = self.train_inputs(batch)
        labels = batch['labels']
        softmax_mask = batch['softmax_mask']
        logits = self(**inputs)
        logits += (softmax_mask - 1) * 1e10

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.nlabels), labels.view(-1))

        ntotal = logits.size(0)
        ncorrect = (logits.argmax(dim=-1) == batch['labels']).long().sum()
        acc = ncorrect / ntotal

        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.train_inputs(batch)
        labels = batch['labels']
        softmax_mask = batch['softmax_mask']
        logits = self(**inputs)
        logits += (softmax_mask - 1) * 1e10

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.nlabels), labels.view(-1))

        ntotal = logits.size(0)
        ncorrect = (logits.argmax(dim=-1) == batch['labels']).long().sum()
        acc = ncorrect / ntotal

        self.log('valid_loss', loss, on_epoch=True, prog_bar=True)
        self.log("valid_acc", acc, on_epoch=True, prog_bar=True)

        return ncorrect, ntotal

    def validation_epoch_end(self, validation_step_outputs):
        ncorrect = 0
        ntotal = 0
        for x in validation_step_outputs:
            ncorrect += x[0]
            ntotal += x[1]
        ncorrect = int(ncorrect.detach().cpu())
        print(f"Validation Accuracy: {round(ncorrect / ntotal, 3)}")


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        paras = list(filter(lambda p: p[1].requires_grad, self.named_parameters()))
        paras = [{
            'params':
            [p for n, p in paras if not any(nd in n for nd in no_decay)],
            'weight_decay':
            0.01
        }, {
            'params': [p for n, p in paras if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]
        optimizer = AdamW(paras, lr=self.hparams.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.total_step * self.hparams.warmup),
            self.total_step)

        return [{
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }]

    def init_model(self, args):
        """
        init function.
        """
        self.pooler_type = args.pooler_type
        self._pooler = Pooler(args.pooler_type)
        if args.pooler_type == "cls":
            self.mlp = MLPLayer(self.hidden_size)
        self.output = OutputLayer(self.hidden_size, self.nlabels)
