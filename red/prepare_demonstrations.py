# -*- coding: utf-8 -*-
# @Time    : 2023/9/16 13:01
# @Author  :
# @Email   :  
# @File    : prepare_demonstrations.py
# @Software: PyCharm
"""
Convert all training instances into demonstrations
with right explain_db, NL rewrite(Dialect) and SQL CoT(Maybe Execution Order).
"""
import json

from main.utils import load_data


def prepare_demonstrations(args):
    # with open(args.table_file) as f:
    #     tables = {ins['db_id']: ins for ins in json.load(f)}

    raw_demos = load_data(args.train_file)
    # Load DB explain
    if args.explain_db:
        with open(args.explain_db, 'r') as f:
            explain_db = json.load(f)
    else:
        explain_db = {}
    # Load NL explain
    if args.explain_nl:
        with open(args.explain_nl, 'r') as f:
            explain_nl_list = json.load(f)
    else:
        explain_nl_list = [""] * len(raw_demos)
    assert len(explain_nl_list) == len(raw_demos), f"explain_nl({len(explain_nl_list)}) must match the demo number"
    # Load SQL explain
    if args.explain_sql:
        with open(args.explain_sql, 'r') as f:
            explain_sql_list = json.load(f)
    else:
        explain_sql_list = [""] * len(raw_demos)
    assert len(explain_sql_list) == len(raw_demos), f"explain_sql({len(explain_sql_list)}) must match the demo number"
    # Assemble explain
    for ins, explain_nl, explain_sql in zip(raw_demos, explain_nl_list, explain_sql_list):
        ins['explain_nl'] = explain_nl
        ins['explain_sql'] = explain_sql
        ins['explain_db'] = explain_db.get(ins['db_id'], {})
        # ins['db_schema'] = tables[ins['db_id']]
    return raw_demos
