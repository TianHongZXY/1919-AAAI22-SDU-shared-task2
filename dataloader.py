# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : dataloader.py
#   Last Modified : 2021-10-17 08:03
#   Describe      : 
#
# ====================================================

import time
import argparse
import itertools
import json
import copy
import os
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from pytorch_lightning import Trainer, seed_everything, loggers
from models.bert_baseline import Bert
from torchsnooper import snoop
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Task2DataModel(pl.LightningDataModule):
    
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Task2DataModel')
        parser.add_argument('--data_dir',
                            default='data/english/legal',
                            type=str)
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--train_data', default='train.json', type=str)
        parser.add_argument('--valid_data', default='dev.json', type=str)
        parser.add_argument('--test_data', default='test.json', type=str)
        parser.add_argument('--diction', default='diction.json')
        #TODO
        parser.add_argument('--ori_diction', default='/home/zxy21/codes_and_data/1919-AAAI22-SDU-shared-task2/data/english/scientific/diction.json')

        parser.add_argument('--cached_train_data',
                            default='cached_train_data.pkl',
                            type=str)
        parser.add_argument('--cached_valid_data',
                            default='cached_valid_data.pkl',
                            type=str)
        parser.add_argument('--cached_test_data',
                            default='cached_test_data.pkl',
                            type=str)
        parser.add_argument('--train_batchsize', default=16, type=int)
        parser.add_argument('--valid_batchsize', default=32, type=int)
        parser.add_argument('--recreate_dataset', action='store_false', default=True)
        
        return parent_args
    
    def __init__(self, args, tokenizer):

        super().__init__()
        self.tokenizer = tokenizer
        self.num_workers = args.num_workers
        self.pretrained_model = args.pretrained_model

        self.cached_data_dir = os.path.join(args.data_dir, args.pretrained_model_name)
        if not os.path.exists(self.cached_data_dir):
            os.makedirs(self.cached_data_dir)

        self.cached_train_data_path = os.path.join(self.cached_data_dir, args.cached_train_data)
        self.cached_valid_data_path = os.path.join(self.cached_data_dir, args.cached_valid_data)
        self.cached_test_data_path = os.path.join(self.cached_data_dir, args.cached_test_data)

        self.train_data_path = os.path.join(args.data_dir, args.train_data)
        self.valid_data_path = os.path.join(args.data_dir, args.valid_data)
        self.test_data_path = os.path.join(args.data_dir, args.test_data)
        self.diction = os.path.join(args.data_dir, args.diction)
        self.ori_diction = json.load(open(args.ori_diction, 'r'))
        self.acronym2lf = json.load(open(self.diction, 'r'))
        self.max_long_form = 0
        self.avg_long_form = 0
        # 统计acronym的最大和平均long form数量
        for acr, lf_list in self.acronym2lf.items():
            self.max_long_form = max(self.max_long_form, len(lf_list))
            self.avg_long_form += len(lf_list)
        self.avg_long_form /= len(self.acronym2lf)
        print(f'In {self.diction}, each acronym has up to {self.max_long_form} long forms and averagely each has {round(self.avg_long_form, 3)} long forms.')

        self.train_batchsize = args.train_batchsize
        self.valid_batchsize = args.valid_batchsize
        # 避免model forward 展平时报错
        assert self.train_batchsize == self.valid_batchsize
        self.recreate_dataset = args.recreate_dataset

        # 把所有acronym的long form list都pad到统一长度, 不过这里用负样本替代[PAD]
        self.acronym2lf_padded = copy.deepcopy(self.acronym2lf)
        print("Creating acronym2lf_padded...")
        for acr, lf_list in tqdm(self.acronym2lf.items()):
            neg_samples = []
            for i in range(self.max_long_form - len(self.acronym2lf[acr])):
                neg_s = random.sample(list(self.acronym2lf.values()), k=1)[0]
                neg_s = random.sample(neg_s, k=1)[0]
                # 避免sample出真样本
                while neg_s in self.acronym2lf[acr]:
                    neg_s = random.sample(list(self.acronym2lf.values()), k=1)[0]
                    neg_s = random.sample(neg_s, k=1)[0]
                neg_samples.append(neg_s)
            self.acronym2lf_padded[acr] += neg_samples
            # 打乱，避免总是分类到前几个位置
            random.shuffle(self.acronym2lf_padded[acr])

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == 'fit':
            self.train_data = self.create_dataset(self.cached_train_data_path,
                                                 self.train_data_path)
            self.valid_data = self.create_dataset(self.cached_valid_data_path,
                                                 self.valid_data_path)
        if stage == 'test':
            self.test_data = self.create_dataset(self.cached_test_data_path,
                                                self.test_data_path,
                                                test=True)

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, collate_fn=self.collate_fn, \
            batch_size=self.train_batchsize, num_workers=self.num_workers, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, collate_fn=self.collate_fn, \
            batch_size=self.valid_batchsize, num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, collate_fn=self.collate_fn, \
            batch_size=self.valid_batchsize, num_workers=self.num_workers, pin_memory=False)

    def create_dataset(self, cached_data_path, data_path, test=False):

        #  if  os.path.exists(cached_data_path) and not self.recreate_dataset:
        if  os.path.exists(cached_data_path):
            print(f'Loading cached dataset from {cached_data_path}...')
            data = torch.load(cached_data_path)
            #  data = list(filter(lambda x: len(self.acronym2lf[x['acronym']]) < 15 and (x['acronym'] in self.ori_diction or random.random() < 0.2), data))
            data = list(filter(lambda x: len(self.acronym2lf[x['acronym']]) < 15, data))
            output = f'In {cached_data_path}, there are {len(data)} instances'
            print(output)
        else:
            print(f'Preprocess {data_path} for Task2...')
            dataset = json.load(open(data_path, 'r'))
            data = []
            avg_long_form_cur = 0
            max_long_form_cur = 0

            for example in tqdm(dataset):
                sentence = example['sentence']
                acronym = example['acronym']
                try:
                    acr_idx = sentence.index(acronym)
                except:
                    print(acronym)
                    print(sentence)
                len_acr = len(acronym)
                sentence = sentence[:acr_idx] + ' [unused1] ' + sentence[acr_idx:acr_idx + len_acr] + ' [unused2] ' + sentence[acr_idx + len_acr:]
                #  print(acronym, sentence)
                #  sentence.insert(acr_idx, '[unused1]')
                #  sentence.insert(acr_idx + 2, '[unused2]')
                #  sentence = ' '.join(sentence)

                max_long_form_cur = max(max_long_form_cur, len(self.acronym2lf[acronym]))
                avg_long_form_cur += len(self.acronym2lf[acronym])
                
                try:
                    encoded = self.tokenizer(self.acronym2lf_padded[acronym], [sentence] * self.max_long_form, padding=True, truncation=True, max_length=512)
                except:
                    print(self.acronym2lf_padded[acronym])
                    print(sentence)
                    raise ValueError()
                sentence = [x + ' [SEP] ' + sentence for x in self.acronym2lf_padded[acronym]]
                input_ids = encoded['input_ids']
                attention_mask = encoded['attention_mask']
                #  for ids in encoded["input_ids"]:
                #      print(tokenizer.decode(ids))
                # 用于预测时mask
                softmax_mask = [0] * self.max_long_form
                for k in range(self.max_long_form):
                    # 真样本的地方是1
                    if(self.acronym2lf_padded[acronym][k] in self.acronym2lf[acronym]):
                        softmax_mask[k] = 1

                # acronym的long_form的索引即为标签
                if not test:
                    long_form = example['label']
                    labels = self.acronym2lf_padded[acronym].index(long_form)

                # Roberta等模型没有token_type_ids
                if 'token_type_ids' not in encoded:
                    encoded['token_type_ids'] = [[0] * len(x) for x in input_ids]
                example = {
                    'idx': example['ID'],
                    'acronym': acronym,
                    'sentence': sentence,
                    'input_ids': torch.LongTensor(input_ids),
                    'attention_mask': torch.LongTensor(attention_mask),
                    'token_type_ids': torch.LongTensor(encoded['token_type_ids']),
                    'softmax_mask': softmax_mask,
                }

                if not test:
                    example['labels'] = labels
                data.append(example)

            avg_long_form_cur /= len(data)
            output = f'In {data_path}, there are {len(data)} instances, each acronym has up to {max_long_form_cur} long forms and averagely each has {round(avg_long_form_cur, 3)} long forms.'
            print(output)
            data = Task2Dataset(data)
            torch.save(data, cached_data_path)
            #  print('example:', example)

        return data

    def collate_fn(self, batch):
        #  print('batch:', batch)
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [example[key] for example in batch]
        #  print('batch_data:', batch_data)
        input_ids = []
        input_ids = list(itertools.chain.from_iterable(batch_data['input_ids']))
        attention_mask = list(itertools.chain.from_iterable(batch_data['attention_mask']))
        token_type_ids = list(itertools.chain.from_iterable(batch_data['token_type_ids']))
        softmax_mask = batch_data['softmax_mask']
        #  softmax_mask = list(itertools.chain.from_iterable(batch_data['softmax_mask']))
        labels = None
        if 'labels' in batch_data:
            labels = torch.LongTensor(batch_data['labels'])

        input_ids = nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        attention_mask = nn.utils.rnn.pad_sequence(attention_mask,
                                                   batch_first=True,
                                                   padding_value=0)
        token_type_ids = nn.utils.rnn.pad_sequence(token_type_ids,
                                                   batch_first=True,
                                                   padding_value=0)

        batch_data = {
            'idx': batch_data['idx'],
            'acronym': batch_data['acronym'],
            'sentence': batch_data['sentence'],
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'softmax_mask': torch.tensor(softmax_mask),
            'labels': labels,
        }

        return batch_data


class Task2Dataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':
    total_parser = argparse.ArgumentParser()

    # * Args for data preprocessing
    total_parser = Task2DataModel.add_data_specific_args(total_parser)
    
    # * Args for training
    #  total_parser = Trainer.add_argparse_args(total_parser)

    # * Args for model specific
    total_parser = Bert.add_model_specific_args(total_parser)

    args = total_parser.parse_args()


    # * Here, we test the data preprocessing
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model,
                                              use_fast=True)

    task2_data = Task2DataModel(args, tokenizer)

    task2_data.setup('fit')
    task2_data.setup('test')

    val_dataloader = task2_data.val_dataloader()

    batch = next(iter(val_dataloader))

    print(batch)
    #  print(batch['softmax_mask'].size())
    #  print(batch['softmax_mask'])


#  def read_data(path):
#      with open(path, 'r') as f:
#          data = f.readlines()
#
#      sent = []
#      abbr = []
#      long_term = []
#      label = []
#
#      for idx, ins in enumerate(data):
#          ins = ins.strip()
#          ins = ins.split('\t')
#          sent.append(ins[0])
#          abbr.append(ins[1])
#          long_term.append(ins[2])
#          label.append(int(ins[3]))
#
#      return sent, abbr, long_term, label
#
#  def preprocess(args, tokenizer):
#      train_sent, train_abbr, train_long_term, train_label = read_data(args.trainset)
#      val_sent, val_abbr, val_long_term, val_label = read_data(args.valset)
#
#      print("Dataset loaded.")
#
#  class CrossInteractionDataset(torch.utils.data.Dataset):
#      def __init__(self, tokenized_data, label):
#          self.input_ids = tokenized_data["input_ids"]
#          self.token_type_ids = tokenized_data["token_type_ids"]
#          self.attention_mask = tokenized_data["attention_mask"]
#          self.label = label
#
#      def __getitem__(self, idx):
#          return {"data":
#                  {"input_ids": self.input_ids[idx],
#                  "token_type_ids": self.token_type_ids[idx],
#                  "attention_mask": self.attention_mask[idx]
#                  },
#                  "label": self.label[idx]
#                  }
#
#      def __len__(self):
#          return len(self.label)
