# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : TianHongZXY
#   Email         : tianhongzxy@163.com
#   File Name     : bert_baseline.py
#   Last Modified : 2021-10-17 08:03
#   Describe      : 
#
# ====================================================

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel
from .base_model import BaseADModel


class Bert(BaseADModel):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer)
        
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
            
        self.tokenizer = tokenizer
        self.nlabels = args.nlabels
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.bert = AutoModel.from_pretrained(args.pretrained_model)
        self.bert.resize_token_embeddings(new_num_tokens=len(tokenizer))
        if not self.hparams.finetune:
            for name, child in self.bert.named_children():
                for param in child.parameters():
                    param.requires_grad = False

        self.init_model(args)

    def forward(self, input_ids, attention_mask, token_type_ids, softmax_mask):
        outputs = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        pooler_output = self._pooler(attention_mask, outputs)
        # If using "cls", we add an extra MLP layer
        # (same as BERT's original implementation) over the representation.
        if self.pooler_type == "cls":
            pooler_output = self.mlp(pooler_output)

        logits = self.output(pooler_output)
        logits = logits.view(self.args.train_batchsize, -1)

        return logits

    def predict(self, input_ids, attention_mask, token_type_ids, softmax_mask):
        outputs = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        pooler_output = self._pooler(attention_mask, outputs)
        if self.pooler_type == "cls":
            pooler_output = self.mlp(pooler_output)

        logits = self.output(pooler_output)
        logits += (softmax_mask - 1) * 1e10

        predict = logits.argmax(dim=-1)
        predict = predict.cpu().tolist()

        return predict
