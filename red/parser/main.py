# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 12:20
# @Author  :
# @Email   :  
# @File    : main.py
# @Software: PyCharm
import json
from copy import deepcopy

from tqdm import tqdm

from red.parser.red_parser import Query
from red.parser.schema import Schema
from red.parser.utils import execute_sql
from eval.bird_evaluator import EvaluateTool as BirdEvaluateTool


if __name__ == '__main__':
    raw_schema = {}
    with open("./datasets/bird/dev_tables.json", 'r') as f:
        raw_schemas = {schema['db_id']: schema for schema in json.load(f)}
    with open("./datasets/bird/dev.json", 'r') as f:
        data = json.load(f)
    with open("./datasets/preds/purple_0613_3072_new.txt") as f:
        preds = f.readlines()
    # with open("./datasets/preds/top_k_pred_bird_7b.txt") as f:
    #     preds = f.readlines()[::9]
    evaluator = BirdEvaluateTool()
    evaluator.register_golds(data, "./datasets/bird/database")
    schema = None
    for idx, ins in enumerate(data):
        test_sql = preds[idx]
        # if idx < 1254: continue
        # test_sql = ins['SQL']
        # score = evaluator.evaluate_one(idx=idx, prediction=test_sql)
        # if idx < 345 or score['exec_match'] == 1:
        # if idx < 0 or ins['SQL'].startswith("WITH") or score['exec_match'] == 1:
        #     continue
        db_id = ins['db_id']
        db_path = f"./datasets/bird/database/{db_id}/{db_id}.sqlite"
        if schema is None or schema.db_id != db_id:
            schema = Schema(raw_schemas[db_id], db_path)
        #
        # if score['exec_match'] == 1:
        #     print("++++++"* 30)
        # else:
        #     print("------" * 30)
        print(f"Instance: {idx}\t{db_id}\n"
              f"[ NL ]\t{ins['question'].strip()}\n"
              f"[KNOW]\t{ins['evidence'].strip()}\n"
              f"[GOLD]\t{ins['SQL'].strip()}\n"
              f"[PRED]\t{test_sql.strip()}")

        flag, values = execute_sql(db_path, test_sql)
        if flag == "exception":
            print(flag, values)
        else:
            q = Query(test_sql, deepcopy(schema))
            try:
                q.validate()
            except ValueError:
                print("ValueError")
        print()
