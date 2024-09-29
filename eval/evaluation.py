# -*- coding: utf-8 -*-
# @Time    : 2023/7/31 09:46
# @Author  :
# @Email   :  
# @File    : evaluation.py
# @Software: PyCharm
import argparse
import json

from eval.spider_evaluator import EvaluateTool as SpiderEvaluateTool
from eval.bird_evaluator import EvaluateTool as BirdEvaluateTool


def main(gold, pred, db_dir, ts_db):
    with open(gold, 'r') as f:
        gold = json.load(f)
    with open(pred, 'r') as f:
        preds = [p.strip() for p in f.readlines()]

    if len(preds[-1]) == 0:
        preds.pop(-1)
    assert len(preds) == len(gold)

    if 'bird' in db_dir:
        evaluator = BirdEvaluateTool(iterate_num=100, meta_time_out=30, verbose=True)
        evaluator.register_golds(gold, db_dir)
    else:
        evaluator = SpiderEvaluateTool(verbose=True)
        evaluator.register_golds(gold, db_dir, ts_db)

    evaluator.evaluate(preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str)
    parser.add_argument("--pred", type=str)
    parser.add_argument("--db", type=str)
    parser.add_argument("--ts_db", default="", type=str)
    args = parser.parse_args()

    main(args.gold, args.pred, args.db, args.ts_db)
