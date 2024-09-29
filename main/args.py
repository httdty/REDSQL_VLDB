# -*- coding: utf-8 -*-
# @Time    : 2023/9/20 16:48
# @Author  :
# @Email   :  
# @File    : args.py
# @Software: PyCharm
import argparse
import datetime
import copy
import os

from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # LLMs config
    parser.add_argument("--model_name",
                        type=str,
                        required=True,
                        help="Model name")
    parser.add_argument("--batch_size",
                        type=int,
                        default=2,
                        help="batch size")

    # Experiment setting
    parser.add_argument("--toy",
                        action="store_true",
                        help="Toy setting for very few instances")
    parser.add_argument("--exp_name",
                        type=str,
                        default="exp",
                        help="Experiment name")
    parser.add_argument("--bug_fix",
                        action="store_true", help="Enable bug fix")
    parser.add_argument("--consistency_num",
                        type=int,
                        default=1,
                        help="consistency size")
    # parser.add_argument("--prompt_length",
    #                     type=int,
    #                     default=2048,
    #                     help="prompt length")
    parser.add_argument("--stage",
                        required=True,
                        choices=["dev", "test"],
                        help="LLMs inference stage: dev for inference with evaluation; test for inference only")
    parser.add_argument("--bug_only",
                        action="store_true", help="Only fix detected bugs")

    # parser.add_argument("--prompt",
    #                     choices=["random", "orange"],
    #                     default="default",
    #                     help="demonstration ranking")

    # RED
    parser.add_argument("--preds",
                        type=str,
                        required=True,
                        help="Top-k prediction file")
    parser.add_argument("--top_k",
                        type=int,
                        help="k number")
    parser.add_argument("--db_content_index_path",
                        type=str,
                        default="",
                        help="db content index path")
    parser.add_argument("--annotation",
                        type=str,
                        default="",
                        help="annotation file path")

    # EXP file path
    parser.add_argument("--output_dir",
                        type=str,
                        default="./output",
                        help="Output dir")
    # parser.add_argument("--train_file",
    #                     type=str,
    #                     default="./datasets/spider/train_spider.json",
    #                     help="train file as demonstrations")
    parser.add_argument("--dev_file",
                        type=str,
                        default="./datasets/spider/dev.json",
                        help="dev file")
    parser.add_argument("--table_file",
                        type=str,
                        default="./datasets/spider/tables.json",
                        help="tables file")
    parser.add_argument("--train_table_file",
                        type=str,
                        default="",
                        help="tables file")
    parser.add_argument("--db_dir",
                        type=str,
                        default="./datasets/spider/database",
                        help="db_dir")
    parser.add_argument("--sample_db_dir",
                        type=str,
                        default="",
                        help="sample_db_dir for test suite")

    args_ = parser.parse_args()
    return args_


def log_args(args_):
    args_dict = vars(copy.deepcopy(args_))
    arg_str = "\n"
    for k, v in args_dict.items():
        if isinstance(v, bool) and v:
            arg_str += f"--{k} "
        else:
            arg_str += f"--{k}={v} "
    logger.info("Running with parameters:" + arg_str)


def output_name(args_):
    args_dict = vars(copy.deepcopy(args_))
    name = args_dict['exp_name']
    keys = [
        'model_name', 'toy', 'consistency_num', 'stage'
    ]
    for k in keys:
        v = args_dict[k]
        if isinstance(v, bool):
            if v:
                name += f"_{k}"
        else:
            name += f"_{k}_{v}"

    name = name.replace(os.sep, '_') + str(datetime.datetime.now()).replace(" ", "_")
    return name


def model_args(args_):
    return {}
