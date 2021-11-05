# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : bad_case_analysis.py
#   Last Modified : 2021-11-05 23:08
#   Describe      : 
#
# ====================================================

import json
import os
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--g", type=str, help="gold file")
    parser.add_argument("--p", type=str, help="prediction file")
    args = parser.parse_args()
    
    gold = json.load(open(args.g, 'r'))
    pred = json.load(open(args.p, 'r'))
    
    bad_cases = []

    for g, p in zip(gold, pred):
        if(p['label'] != g['label']):
            w_case = g
            g['predicted'] = p['label']
            bad_cases.append(w_case)
    json.dump(bad_cases, open(os.path.join(os.path.split(args.p)[0], "bad_cases.json"), 'w'), indent=4)
    
