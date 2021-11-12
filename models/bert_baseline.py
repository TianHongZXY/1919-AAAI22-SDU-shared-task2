# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : TianHongZXY
#   Email         : tianhongzxy@163.com
#   File Name     : bert_baseline.py
#   Last Modified : 2021-11-07 17:39
#   Describe      : 
#
# ====================================================

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel, AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from .base_model import BaseADModel
from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap
from torchsnooper import snoop
from ChildTuningOptimizer import ChildTuningAdamW


class Bert(BaseADModel):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer)
        
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        self.bert = AutoModel.from_pretrained(self.args.pretrained_model)
        self.bert.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        if not self.hparams.finetune:
            for name, child in self.bert.named_children():
                for param in child.parameters():
                    param.requires_grad = False

    def adv_forward(self, logits, input_ids, attention_mask, token_type_ids):
        adv_loss = self.adv_loss_fn(model=self,
                                    logits=logits,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    input_ids=input_ids)

        adv_loss = self.hparams.adv_alpha * adv_loss

        return adv_loss

    def forward(self, attention_mask, token_type_ids, softmax_mask=None, input_ids=None, inputs_embeds=None):
        if inputs_embeds is not None:
            outputs = self.bert(attention_mask=attention_mask, token_type_ids=token_type_ids, 
                                inputs_embeds=inputs_embeds, output_hidden_states=True, return_dict=False)
        else:
            outputs = self.bert(input_ids, attention_mask, token_type_ids, 
                                output_hidden_states=True, return_dict=False)

        pooler_output = self._pooler(attention_mask, outputs)
        # If using "cls", we add an extra MLP layer
        # (same as BERT's original implementation) over the representation.
        if self.pooler_type == "cls":
            pooler_output = self.mlp(pooler_output)

        logits = self.output(pooler_output)
        logits = logits.view(-1, self.args.nlabels)

        return logits

    def predict(self, input_ids, attention_mask, token_type_ids, softmax_mask):
        logits = self(input_ids=input_ids, 
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids, 
                      )

        logits += (softmax_mask - 1) * 1e10

        predict = logits.argmax(dim=-1)
        predict = predict.cpu().tolist()

        return predict

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_paras = list(self.bert.named_parameters())
        bert_paras = [
            {'params': [p for n, p in bert_paras if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': self.hparams.bert_lr},
            {'params': [p for n, p in bert_paras if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': self.hparams.bert_lr}
        ]
        
        named_paras = list(self.named_parameters())
        head_paras = [
            {'params': [p for n, p in named_paras if 'bert' not in n], 'lr': self.hparams.lr}
        ]

        paras = bert_paras + head_paras

        if self.hparams.child_tuning:
            optimizer = ChildTuningAdamW(paras, lr=self.hparams.lr)
        else:
            optimizer = AdamW(paras, lr=self.hparams.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(self.total_step * self.hparams.warmup), self.total_step)

        return [
            {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        ]

    #  def configure_sharded_model(self):
    #      self.mlp = auto_wrap(self.mlp)
    #      self.output = auto_wrap(self.output)
    #      self._pooler = auto_wrap(self._pooler)
    #      #  self.bert = auto_wrap(self.bert)
    #      self.model = nn.Sequential(self.mlp, self.output, self._pooler, self.bert)


