# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 21:44
# @Author  :
# @Email   :  
# @File    : utils.py
# @Software: PyCharm
import copy
import json
import os.path
import sqlite3
from typing import Dict, List
import pickle


def toy(ori_list):
    return ori_list[1400::5]


def package_sql(preds: List[str], db_ids: List[str]):
    res = {}
    for i, (pred, db_id) in enumerate(zip(preds, db_ids)):
        res[f'{i}'] = f"{pred}\t----- bird -----\t{db_id}"
    return res


def load_data(data_file: str):
    with open(data_file, 'r') as f:
        data = json.load(f)
    idx = 0
    for ex in data:
        ex['idx'] = idx
        idx += 1
    return data


def load_skeleton(skeleton_file: str):
    with open(skeleton_file, 'r') as f:
        pred = json.load(f)
    pred = [p[0]['generated_text'] for p in pred]
    return pred


def load_domain(domain_file: str):
    with open(domain_file, 'r') as f:
        domain = json.load(f)
    return domain


def load_processed_data(data_file: str):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data


def load_schema(tables_file: str, db_dir: str) -> Dict:
    with open(tables_file, 'r') as f:
        schemas = json.load(f)
    schema = {}
    for s in schemas:
        s['description'] = "; ".join(db_info(s))
        s['description_table'] = "\n# ".join(db_info_table(s))
        s['description_content'] = "\n".join(db_info_content(s, db_dir))
        schema[s['db_id']] = copy.deepcopy(s)
    return schema


def db_info(schema):
    """ Generate description of database.

    Args:
        schema: description of database schema

    Returns:
        list of description lines
    """
    lines = []
    # amend_missing_foreign_keys(schema)
    tables = schema['table_names_original']
    all_columns = schema['column_names_original']
    pks = schema['primary_keys']
    fks = schema['foreign_keys']
    nr_tables = len(tables)
    for tbl_idx in range(nr_tables):
        tbl_name = tables[tbl_idx]
        tbl_name = tbl_name.lower()

        # table_columns = [c[1] for c in all_columns if c[0] == tbl_idx]
        table_columns = [
            c[1] + "(primary key)" if i in pks else c[1]
            for i, c in enumerate(all_columns) if c[0] == tbl_idx
        ]
        table_columns = [c.lower() for c in table_columns]
        quoted_columns = ["'" + c + "'" for c in table_columns]
        col_list = ', '.join(quoted_columns)
        line = f"Table '{tbl_name}' with columns {col_list}"
        lines.append(line)
    fk_hint = ["Foreign keys are:"]
    for fk in fks:
        fkl, fkr = fk
        fkl = tables[all_columns[fkl][0]] + "." + all_columns[fkl][1]
        fkr = tables[all_columns[fkr][0]] + "." + all_columns[fkr][1]
        fk_hint.append(f"({fkl}, {fkr})".lower())
    lines.append(" ".join(fk_hint))
    return lines


def db_info_table(schema):
    """ Generate description of database as Tab(col, col,...) type.

    Args:
        schema: description of database schema

    Returns:
        list of description lines
    """
    lines = []
    tables = schema['table_names_original']
    all_columns = schema['column_names_original']
    nr_tables = len(tables)
    for tbl_idx in range(nr_tables):
        tbl_name = tables[tbl_idx]

        table_columns = [c[1] for c in all_columns if c[0] == tbl_idx]
        col_list = ', '.join(table_columns)
        line = f"{tbl_name}({col_list})"
        lines.append(line)
    return lines


def db_info_content(schema, db_dir):
    """ Generate description of database as
    SELECT * FROM tab LIMIT 3;
    Table: xxx
    xxx  xxx  xxx
    xxx  xxx  xxx
    xxx  xxx  xxx
    xxx  xxx  xxx

    Args:
        schema: description of database schema
        db_dir: database dir path

    Returns:
        list of description lines
    """
    lines = []
    tables = schema['table_names_original']
    db_file = os.path.join(db_dir, schema['db_id'], f"{schema['db_id']}.sqlite")
    conn = sqlite3.connect(db_file)
    for tab in tables:
        if tab == "sqlite_sequence":
            continue
        sql = f"SELECT * FROM {tab} LIMIT 3"
        cursor = conn.cursor()

        # fetch table values
        cursor.execute(sql)
        col_head = '\t'.join([d[0] for d in cursor.description])
        rows = []
        for row in cursor.fetchall():
            rows.append([str(r) for r in row])
        rows = "\n".join(["\t".join(row) for row in rows])
        line = f"3 example rows from table {tab}:\n" \
               f"{sql}\n" \
               f"Table: {tab}\n" \
               f"{col_head}\n" \
               f"{rows}\n"
        lines.append(line)
    return lines


# def clean_output(result, mapping, preds: List[str]):
#     if not preds:
#         return result
#     mapping = json.loads(mapping)
#
#     count = {k: 0 for k in mapping}
#     max_key = list(mapping.keys())[0]
#     for pred in preds:
#         for key in mapping:
#             if key in pred.split("\n")[0]:
#                 count[key] += 1
#                 if count[key] > count[max_key]:
#                     max_key = key
#
#     return mapping[max_key]


def parse_json(json_str):
    json_str = json_str.strip().strip("```").strip().strip("json").strip()
    json_str = json_str.replace("\n", " ")
    json_str = json_str.replace(r"\'", "'")
    json_str = json_str.replace(r'\"', "'")
    json_str = json_str.replace(r"\\", "")
    json_str = json_str[json_str.index("{"): json_str.rindex("}") + 1]
    res_dict = json.loads(json_str)
    return str(res_dict['SQL'])


def clean_output(result, preds: List[str]):
    res = []
    if not preds:
        return [result]
    for pred in preds:
        try:
            pred = parse_json(pred)
        except Exception as e:
            print(e, pred)
            if "SQL" in pred:
                pred = pred[pred.index("SQL"):]
            if "select" in pred.lower():
                pred = pred[pred.lower().index("select"):].split("\n", maxsplit=1)[0]
            pred = pred.strip().strip('"').strip("'").strip()

        pred = pred.strip().replace("\n", " ")
        # while "  " in pred:
        #     pred = pred.replace("  ", " ")
        # print(pred)
        if pred.lower().strip().startswith("```sql"):
            pred = pred.strip('```')
            pred = pred.lstrip('sql')
            pred = pred.strip()
        if 'select' in pred.lower():
            if ": select" in pred.lower():
                pred = pred[pred.lower().index(": select") + 2:]
            elif "= select" in pred.lower():
                pred = pred[pred.lower().index("= select") + 2:]
            else:
                pred = pred[pred.lower().index("select"):]
        else:
            pred = "SELECT * from `table`;"
        pred = pred.split(";")[0] + ";"
        pred = pred.replace("\n", " ").strip()
        # pred = pred.split("\n")[0].strip()
        if not pred.strip().lower().startswith("select"):
            pred = "SELECT " + pred.strip()

        pred = pred.strip()
        pred = pred.replace('<>', '!=')
        if pred.startswith('1.'):
            pred = pred.split('1.')[1].strip()

        # if not pred.upper().startswith('SELECT'):
        #     pred = "SELECT " + pred

        # if 'Comment' in pred:
        #     pred = pred.split('Comment')[0] + "\n"
        res.append(pred)

    return res
