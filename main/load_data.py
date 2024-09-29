# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 19:47
# @Author  :
# @Email   :  
# @File    : load_data.py
# @Software: PyCharm
import json
import os

from tqdm import tqdm

from main.utils import load_data
from red.red import RED


def load_gold_data(args):
    dev = load_data(args.dev_file)
    for ins in dev:
        if 'query' not in ins:
            ins['query'] = ins['SQL']
    return dev


def data(args):
    # Load data
    dev = load_data(args.dev_file)

    # Load candidates
    with open(args.preds, 'r') as f:
        preds = [p.strip() for p in f.readlines() if p.strip()]
    # all_k = len(preds) // len(dev)
    # assert len(preds) == len(dev) * all_k, "Please keep candidate num same"
    assert len(preds) == len(dev), "Please keep candidate num same"
    db_qa = {}
    # db_id2searcher = {}
    if args.annotation:
        with open(args.annotation, 'r') as f:
            annotations = json.load(f)
    else:
        annotations = None

    for i in range(len(dev)):
        dev[i]['pred'] = preds[i]
        db_id = dev[i]['db_id']
        if db_id not in db_qa:
            db_qa[db_id] = []
        db_qa[db_id].append([dev[i]['question'], preds[i]])

        # BM25 Index
        if args.db_content_index_path:
            # if db_id not in db_id2searcher:
            #     db_id2searcher[db_id] = LuceneSearcher(os.path.join(str(args.db_content_index_path), db_id))
            dev[i]['index'] = os.path.join(str(args.db_content_index_path), db_id)
        else:
            dev[i]['index'] = None

        if annotations:
            dev[i]['annotation'] = annotations[db_id]
        else:
            dev[i]['annotation'] = None
    return dev

    # # Init RED
    # red = RED(args.train_table_file, args.table_file, args.db_dir)
    #
    # # Prompt gen
    # for ins in tqdm(dev):
    #     ins['prompt'] = red.refine(ins)
    #
    # # Data prepare
    # instances = []
    # for ins in dev:
    #     instances.append(
    #         {
    #             "prompt": ins['prompt'],
    #             "pre_pred": ins['pred'],
    #             "db_id": ins['db_id'],
    #         }
    #     )
    #
    # return instances
