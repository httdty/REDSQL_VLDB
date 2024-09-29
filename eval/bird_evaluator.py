# -*- coding: utf-8 -*-
# @Time    : 2024/1/31 17:01
# @Author  :
# @Email   :  
# @File    : evaluator.py
# @Software: PyCharm
# encoding=utf8
import pickle
import sqlite3
import sys

from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm

from eval.bird_evaluation.evaluation_acc import print_data as acc_print
from eval.bird_evaluation.evaluation_ves import execute_sql, clean_abnormal, print_data as ves_print, compute_ves

# load a pre-computed results for dev or test:
# pre_results = pickle.load(open('./bird/dev_result.bin', 'rb'))

import os
from typing import List


def iterated_execute_sql(predicted_sql, ground_truth, db_path, iterate_num):
    conn = sqlite3.connect(db_path)
    diff_list = []
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    time_ratio = 0
    acc = 0
    if set(predicted_res) == set(ground_truth_res):
        acc = 1
        for i in range(iterate_num):
            predicted_time = execute_sql(predicted_sql, db_path)
            ground_truth_time = execute_sql(ground_truth, db_path)
            diff_list.append(ground_truth_time / predicted_time)
        processed_diff_list = clean_abnormal(diff_list)
        time_ratio = sum(processed_diff_list) / len(processed_diff_list)
    return acc, time_ratio


def execute_model(predicted_sql, ground_truth, db_place, idx, iterate_num, meta_time_out):
    try:
        # you can personalize the total timeout number
        # larger timeout leads to more stable ves
        # while it needs more your patience....
        acc, time_ratio = func_timeout(meta_time_out * iterate_num, iterated_execute_sql,
                                       args=(predicted_sql, ground_truth, db_place, iterate_num))
        # print([idx, math.sqrt(time_ratio)])
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        # result = [(f'timeout',)]
        time_ratio = 0
        acc = 0
    except Exception as e:
        # result = [(f'error',)]  # possibly len(query) > 512 or not executable
        time_ratio = 0
        acc = 0
    result = {'sql_idx': idx, 'acc': acc, 'time_ratio': time_ratio}
    return result


class EvaluateTool(object):
    def __init__(self, iterate_num=100, meta_time_out=30, verbose=False, **kwargs):
        # self.args = args
        self.golds: List[dict] = []
        self.iterate_num = iterate_num
        self.meta_time_out = meta_time_out
        self.verbose = verbose
        self.kwargs = kwargs
        self.exec_result = []

    def register_golds(self, dataset, db_path):
        for idx, sample in enumerate(dataset):
            db_id = sample["db_id"]

            self.golds.append(
                {
                    "query": sample["SQL"],
                    "question": sample["question"],
                    "evidence": sample["evidence"],
                    "db_id": db_id,
                    "db_path": db_path,
                    "difficulty": sample['difficulty'],
                }
            )

    def evaluate(self, preds):
        res = []
        for idx, pred in tqdm(enumerate(preds)):
            res.append(self.evaluate_one(idx, pred))
        self.print_score()
        return res

    def evaluate_one(self, idx, prediction):
        reference = self.golds[idx]
        db_path = os.path.join(reference['db_path'], reference['db_id'], reference['db_id'] + ".sqlite")
        res = execute_model(prediction, reference['query'], db_path, idx, iterate_num=self.iterate_num,
                            meta_time_out=self.meta_time_out)
        self.exec_result.append(res)

        return {
            "exact_match": 0,
            "exec_match": res['acc'],
            "test_suite_match": res['time_ratio'],
        }

    def print_score(self):
        print("################################################ Bird ACC Evaluation "
              "################################################")
        simple_acc, moderate_acc, challenging_acc, acc, count_lists = self.compute_acc_by_diff()
        score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
        acc_print(score_lists, count_lists)

        print("\n################################################ Bird VES Evaluation "
              "################################################")
        simple_ves, moderate_ves, challenging_ves, ves, count_lists = self.compute_ves_by_diff()
        score_lists = [simple_ves, moderate_ves, challenging_ves, ves]
        ves_print(score_lists, count_lists)

    def compute_acc_by_diff(self):
        num_queries = len(self.exec_result)
        results = [res['acc'] for res in self.exec_result]
        simple_results, moderate_results, challenging_results = [], [], []

        for i, content in enumerate(self.golds):
            if content['difficulty'] == 'simple':
                simple_results.append(self.exec_result[i])

            if content['difficulty'] == 'moderate':
                moderate_results.append(self.exec_result[i])

            if content['difficulty'] == 'challenging':
                challenging_results.append(self.exec_result[i])

        simple_acc = sum([res['acc'] for res in simple_results]) / len(simple_results)
        moderate_acc = sum([res['acc'] for res in moderate_results]) / len(moderate_results)
        challenging_acc = sum([res['acc'] for res in challenging_results]) / len(challenging_results)
        all_acc = sum(results) / num_queries
        count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
        return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists

    def compute_ves_by_diff(self):
        num_queries = len(self.exec_result)
        simple_results, moderate_results, challenging_results = [], [], []
        for i, content in enumerate(self.golds):
            if content['difficulty'] == 'simple':
                simple_results.append(self.exec_result[i])
            if content['difficulty'] == 'moderate':
                moderate_results.append(self.exec_result[i])
            if content['difficulty'] == 'challenging':
                challenging_results.append(self.exec_result[i])
        simple_ves = compute_ves(simple_results)
        moderate_ves = compute_ves(moderate_results)
        challenging_ves = compute_ves(challenging_results)
        all_ves = compute_ves(self.exec_result)
        count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
        return simple_ves, moderate_ves, challenging_ves, all_ves, count_lists
