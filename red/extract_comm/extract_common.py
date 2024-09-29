from typing import List

from red.extract_comm.process_sql import Schema, get_schema, get_sql_loc, tokenize
from red.extract_comm.evaluation import build_foreign_key_map, build_valid_col_units, rebuild_sql_col, rebuild_sql_val
from red.extract_comm.evaluation import eval_sel, eval_where, eval_group, eval_having, eval_and_or, eval_order, eval_IUEN, eval_from
from red.extract_comm.evaluation import eval_keywords, get_scores
from red.extract_comm.process import get_sql
import json
import os


def get_topk(topk_path):
    topk_list = []
    tmp = []

    with open(topk_path, 'r') as f:
        for line in f:
            if (line.strip()):
                line = line.strip()
                tmp.append(line)
            elif tmp:
                topk_list.append(tmp)
                tmp = []
    return topk_list


def eval_partial_match(s1, s2, del_not):
    s1_total, s2_total, cnt = eval_from(s1, s2)
    acc, rec, f1 = get_scores(cnt, s1_total, s2_total)
    if acc == 0:
        del_not['from'] = True

    s1_total, s2_total, cnt, cnt_wo_agg = eval_sel(s1, s2)
    acc, rec, f1 = get_scores(cnt, s1_total, s2_total)
    acc_, rec, f1 = get_scores(cnt_wo_agg, s1_total, s2_total)
    if acc == 0:
        del_not['select'] = True

    s1_total, s2_total, cnt, cnt_wo_agg = eval_where(s1, s2)
    acc, rec, f1 = get_scores(cnt, s1_total, s2_total)
    acc_, rec, f1 = get_scores(cnt_wo_agg, s1_total, s2_total)
    if acc == 0:
        del_not['where'] = True

    s1_total, s2_total, cnt = eval_group(s1, s2)
    acc, rec, f1 = get_scores(cnt, s1_total, s2_total)
    if acc == 0:
        del_not['groupBy'] = True

    s1_total, s2_total, cnt = eval_having(s1, s2)
    acc, rec, f1 = get_scores(cnt, s1_total, s2_total)
    if acc == 0:
        del_not['having'] = True

    s1_total, s2_total, cnt = eval_order(s1, s2)
    acc, rec, f1 = get_scores(cnt, s1_total, s2_total)
    if acc == 0:
        del_not['orderBy'] = True

    s1_total, s2_total, cnt = eval_and_or(s1, s2)
    acc, rec, f1 = get_scores(cnt, s1_total, s2_total)

    s1_total, s2_total, cnt = eval_IUEN(s1, s2)
    acc, rec, f1 = get_scores(cnt, s1_total, s2_total)

    s1_total, s2_total, cnt = eval_keywords(s1, s2)
    acc, rec, f1 = get_scores(cnt, s1_total, s2_total)

    return del_not


def del_dif(sql, idx_list, del_not):
    toks = tokenize(sql)

    if del_not['from']:
        a = idx_list['from'][0]
        b = idx_list['from'][1]
        for i in range(a, b):
            toks[i] = ' '
    if del_not['select']:
        a = idx_list['select'][0]
        b = idx_list['select'][1]
        for i in range(a, b):
            toks[i] = ' '
    if del_not['where']:
        a = idx_list['where'][0]
        b = idx_list['where'][1]
        for i in range(a, b):
            toks[i] = ' '
    if del_not['groupBy']:
        a = idx_list['groupBy'][0]
        b = idx_list['groupBy'][1]
        for i in range(a, b):
            toks[i] = ' '
    if del_not['having']:
        a = idx_list['having'][0]
        b = idx_list['having'][1]
        for i in range(a, b):
            toks[i] = ' '
    if del_not['orderBy']:
        a = idx_list['orderBy'][0]
        b = idx_list['orderBy'][1]
        for i in range(a, b):
            toks[i] = ' '
    return toks


def eval_topk(s1, s2, tables, db_dir, schema, del_not):
    s1_valid_col_units = build_valid_col_units(s1["from"]["table_units"], schema)
    s1 = rebuild_sql_val(s1)
    s1 = rebuild_sql_col(s1_valid_col_units, s1, tables)
    s2_valid_col_units = build_valid_col_units(s2["from"]["table_units"], schema)
    s2 = rebuild_sql_val(s2)
    s2 = rebuild_sql_col(s2_valid_col_units, s2, tables)
    del_not = eval_partial_match(s1, s2, del_not)
    return del_not


def etract_common_parts(topk, tables, db_dir, schema):
    del_not = {
        "except": False,
        "from": False,
        "groupBy": False,
        "having": False,
        "intersect": False,
        "limit": False,
        "orderBy": False,
        "select": False,
        "union": False,
        "where": False,
    }
    len_ = len(topk)
    for i in range(len_):
        for j in range(len_):
            if i >= j:
                continue
            del_not = eval_topk(topk[i], topk[j], tables, db_dir, schema, del_not)
    return del_not


def sql_nested_query_tmp_name_convert(sql: str, nested_level=0, sub_query_token='S') -> str:
    sql = sql.replace('(', ' ( ')
    sql = sql.replace(')', ' ) ')
    tokens = sql.split()
    select_count = sql.lower().split().count('select')
    level_flag = sub_query_token * nested_level

    # recursive exit
    if select_count == 1:
        # need to fix the last level's tmp name
        res = sql
        if nested_level:
            # log all tmp name
            tmp_name_list = set()
            for i in range(len(tokens)):
                # find tmp name
                if tokens[i].lower() == 'as':
                    tmp_name_list.add(tokens[i + 1])
                # convert every tmp name
            for tmp_name in tmp_name_list:
                res = res.replace(f' {tmp_name}', f' {level_flag}{tmp_name}')
        return res

    # for new sql token
    new_tokens = list()
    bracket_num = 0
    i = 0
    # iter every token in tokens
    while i < len(tokens):
        # append ordinary token
        new_tokens.append(tokens[i])
        # find a nested query
        if tokens[i] == '(' and tokens[i + 1].lower() == 'select':
            nested_query = ''
            bracket_num += 1
            left_bracket_position = i + 1
            # in one nested query
            while bracket_num:
                i += 1
                if tokens[i] == '(':
                    bracket_num += 1
                elif tokens[i] == ')':
                    bracket_num -= 1
                # to the end of the query
                if bracket_num == 0:
                    # format new nested query and get the tokens
                    nested_query = ' '.join(tokens[left_bracket_position: i])
                    nested_query = sql_nested_query_tmp_name_convert(nested_query, nested_level + 1)
            # new sql token log
            new_tokens.append(nested_query)
            # append the right bracket
            new_tokens.append(tokens[i])
        # IUE handle
        elif tokens[i].lower() in {'intersect', 'union', 'except'}:
            nested_query = ' '.join(tokens[i + 1:])
            nested_query = sql_nested_query_tmp_name_convert(nested_query, nested_level + 10)
            new_tokens.append(nested_query)
            i += 9999
        i += 1
    # format the new query
    res = ' '.join(new_tokens)
    if nested_level:
        # log all tmp name
        tmp_name_list = set()
        for i in range(len(new_tokens)):
            # find tmp name
            if new_tokens[i].lower() == 'as':
                tmp_name_list.add(new_tokens[i + 1])
            # convert every tmp name
        for tmp_name in tmp_name_list:
            res = res.replace(f' {tmp_name}', f' {level_flag}{tmp_name}')

    return res


def extract(db_name: str, db_dir: str, tab_path: str, topk_list: List[str]) -> str:
    schema = {}
    db_path = os.path.join(db_dir, db_name, db_name + ".sqlite")
    schema[db_name] = Schema(get_schema(db_path))

    with open(tab_path, 'r') as f:
        data = json.load(f)
    tables = {}
    for entry in data:
        tables[entry["db_id"]] = build_foreign_key_map(entry)

    parsed_sql_list = []
    tmp = []
    cnt = 0
    for i, query in enumerate(topk_list):
        query = sql_nested_query_tmp_name_convert(query)
        if i == 0:
            idx_list = {
                "except": [],
                "from": {"conds": [], "table_units": []},
                "groupBy": [],
                "having": [],
                "intersect": [],
                "limit": [],
                "orderBy": [],
                "select": [],
                "union": [],
                "where": [],
            }
            idx_list, sql = get_sql_loc(schema[db_name], query, idx_list)
        else:
            sql = get_sql(schema[db_name], query)
        tmp.append(sql)
    del_not = etract_common_parts(tmp, tables, db_dir, schema[db_name])
    toks = del_dif(topk_list[0], idx_list, del_not)
    sql = ' '.join([s for s in toks if s != ' '])
    sql = sql.replace(" ( ", "(").replace(" ) ", ") ").replace(" , ", ", ")
    return sql
