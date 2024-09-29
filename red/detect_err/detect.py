import sqlite3
import os
import re
import json

from eval.spider.process_sql import get_schema, Schema, get_tables_with_alias, tokenize
from main.preprocessing import normalization


def sql_nested_query_tmp_name_convert(sql: str, nested_level=0, sub_query_token='S') -> str:
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


def build_fk_map(entry):
    cols_orig = entry["column_names_original"]
    tables_orig = entry["table_names_original"]

    # rebuild cols corresponding to idmap in Schema
    cols = []
    for col_orig in cols_orig:
        if col_orig[0] >= 0:
            t = tables_orig[col_orig[0]]
            c = col_orig[1]
            cols.append(t.lower() + "." + c.lower())
        else:
            cols.append("__all__")

    def keyset_in_list(k1, k2, k_list):
        for k_set in k_list:
            if k1 in k_set or k2 in k_set:
                return k_set
        new_k_set = set()
        k_list.append(new_k_set)
        return new_k_set

    foreign_key_list = []
    foreign_keys = entry["foreign_keys"]
    for fkey in foreign_keys:
        key1, key2 = fkey
        key_set = keyset_in_list(key1, key2, foreign_key_list)
        key_set.add(key1)
        key_set.add(key2)

    foreign_key_map = {}
    for key_set in foreign_key_list:
        sorted_list = sorted(list(key_set))
        midx = sorted_list[0]
        for idx in sorted_list:
            foreign_key_map[cols[idx]] = cols[midx]

    return foreign_key_map


def build_fk_map_from_json(table):
    with open(table) as f:
        data = json.load(f)
    tables = {}
    for entry in data:
        tables[entry["db_id"]] = build_fk_map(entry)
    return tables


def build_primary_key_list(entry):
    cols_orig = entry["column_names_original"]
    primary_key_list = []
    primary_keys = entry["primary_keys"]
    for i in primary_keys:
        primary_key_list.append(cols_orig[i][1])
    return primary_key_list


def build_primary_key_list_from_json(table: str):
    with open(table) as f:
        data = json.load(f)
    tables = {}
    for entry in data:
        tables[entry["db_id"]] = build_primary_key_list(entry)
    return tables


def build_column_table_map(entry):
    cols_orig = entry["column_names_original"]
    tables_orig = entry["table_names_original"]
    col_tab_map = {}
    for item in cols_orig:
        if item[1] == "*":
            col_tab_map["*"] = "__all__"
        else:
            col_tab_map[item[1].lower()] = tables_orig[item[0]]
    return col_tab_map


def build_column_table_map_from_json(table: str):
    with open(table) as f:
        data = json.load(f)
    tables = {}
    for entry in data:
        tables[entry["db_id"]] = build_column_table_map(entry)
    return tables


def isprimary(col, db_name, pks):
    for col_name in pks:
        if col_name == col:
            return True
    return False


def groupby_fix(sql, db_name, pks):
    pattern = r'group by (\w+)'
    match = re.search(pattern, sql.lower())
    if match:
        if "join" in sql.lower():
            return sql
        elif sql.lower().count("select") > 1:
            return sql
        else:
            col = match.group(1)
            if isprimary(col, db_name, pks):
                sql = sql.replace(f' {col}', "")
    return sql


def process(pair, tables_with_alias):
    tmp = []
    for item in pair:
        parts = item.split(".")
        col = parts[1]
        table = tables_with_alias[parts[0].lower()]
        tmp.append(table.lower() + "." + col.lower())

    return (tmp[0], tmp[1])


def check_fk(pairs, fks, db_name, tables_with_alias):
    for pair in pairs:
        pair = process(pair, tables_with_alias)
        if pair[0] not in fks or pair[1] not in fks:
            return False
        if fks[pair[0]] == pair[1] or fks[pair[1]] == pair[0]:
            return True
    return False


def post_process(sql, alias):
    for item in alias:
        sql = sql.replace(f'{item}.', "")
    return sql


def join_fix(sql, fks, db_name, tables_with_alias):
    alias = []
    if "from" in sql:
        tmp = sql.split("from")
    else:
        tmp = sql.split("FROM")
    from_clauses = []
    for i in range(1, len(tmp)):
        from_clauses.append("from" + tmp[i])
    result_clauses = []
    result_clauses.append(tmp[0])
    for i in range(len(from_clauses)):
        parts = from_clauses[i].split()
        if "join" in from_clauses[i].lower():
            pattern = r"\bon\b"
            matches = re.finditer(pattern, from_clauses[i].lower())
            pairs = []
            for match in matches:
                idx = match.end()
                words = from_clauses[i][idx:].split()
                pairs.append((words[0], words[2]))
            if not check_fk(pairs, fks, db_name, tables_with_alias):
                while "on" in parts or "ON" in parts:
                    if "on" in parts:
                        idx = parts.index("on") + 4
                    else:
                        idx = parts.index("ON") + 4
                    for j in range(1, idx):
                        if parts[j].lower() == "as":
                            alias.append(parts[j + 1])
                        parts[j] = ""
        temp = " ".join(parts)
        result_clauses.append(temp)

    sql = " ".join(result_clauses)
    sql = ' '.join(sql.split())
    sql = post_process(sql, alias)
    return sql


def check_value(val, col, table, db_path):
    if val.isdigit():
        return True
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    sql = f"SELECT COUNT(*) FROM {table} WHERE {col} = '{val}'"
    cursor.execute(sql)
    num = cursor.fetchall()
    if num[0][0] == 0:
        return False
    return True


def value_fix(sql, c2t, db_name, tables_with_alias, db_path):
    pattern = r"\bwhere\b"
    matches = re.finditer(pattern, sql.lower())
    vals = []
    cols = []
    for match in matches:
        idx = match.end()
        words = sql[idx:].split()
        if len(words) < 3:
            return sql
        idx = 0
        cols.append(words[idx])
        if words[idx + 2] == '(':
            return sql
        if words[idx + 2].startswith("'") or words[idx + 2].startswith("\""):
            tmpval = []
            tmpval.append(words[idx + 2])
            idx += 2
            if not (words[idx].endswith("'") or words[idx].endswith("\"") or words[idx].endswith("\")") or words[
                idx].endswith("\')")):
                idx += 1
                while not (words[idx].endswith("'") or words[idx].endswith("\"") or words[idx].endswith("\")") or words[
                    idx].endswith("\')")):
                    tmpval.append(words[idx])
                    idx += 1
                tmpval.append(words[idx])
                idx += 1
                vals.append(" ".join(tmpval))
            else:
                vals.append(" ".join(tmpval))
                idx += 3

        if words[1] == 'between':
            cols.append(words[0])
            vals.append(words[idx + 2])
        elif len(words) - idx > 3:
            if words[idx].lower() == "or" or words[idx].lower() == "and":
                cols.append(words[idx + 1])
                if words[idx + 3].startswith("'") or words[idx + 3].startswith("\""):
                    tmpval = []
                    tmpval.append(words[idx + 3])
                    idx += 3
                    if not (words[idx].endswith("'") or words[idx].endswith("\"")):
                        idx += 1
                        while not (words[idx].endswith("'") or words[idx].endswith("\"")):
                            tmpval.append(words[idx])
                            idx += 1
                        tmpval.append(words[idx])
                        idx += 1
                        vals.append(" ".join(tmpval))
                    else:
                        vals.append(" ".join(tmpval))

    for i in range(len(vals)):
        val = vals[i]
        col = cols[i]
        val = val.replace("'", "")
        if "." in col:
            parts = col.split(".")
            col = parts[1]
            table = tables_with_alias[parts[0].lower()]
        elif "join" not in sql.lower():
            table = find_nearest_from(val, sql.split())
        else:
            table = c2t[db_name][col.lower()]
        if not check_value(val, col, table, db_path):
            sql = sql.replace(val, "")
    return sql


def find_nearest_from(val, lst):
    found_from = None
    for i in range(len(lst)):
        if lst[i].lower() == 'from' and i < len(lst) - 1:
            found_from = lst[i + 1]
            break
    return found_from


def orderby_fix(sql):
    if "select" in sql:
        tmp = sql.split("select")
    else:
        tmp = sql.split("SELECT")
    select_clauses = []
    for i in range(1, len(tmp)):
        select_clauses.append("select" + tmp[i])
    result_clauses = []
    for i in range(len(select_clauses)):
        parts = select_clauses[i].split()
        if any(agg_func in select_clauses[i].lower() for agg_func in ["max(", "min(", "sum(", "avg("]):
            if "order" in select_clauses[i].lower():
                oidx = select_clauses[i].lower().split().index("order")
                fidx = select_clauses[i].lower().split().index("from")
                prefixes = ["max(", "min(", "sum(", "avg("]
                for j in range(1, fidx):
                    for prefix in prefixes:
                        if parts[j].lower().startswith(prefix):
                            parts[j] = ""
                for j in range(oidx + 2, oidx + 4):
                    parts[j] = ""
        temp = " ".join(parts)
        result_clauses.append(temp)
    sql = " ".join(result_clauses)
    sql = ' '.join(sql.split())
    return sql


def error_fix(db_name: str, db_dir: str, tables: str, sql: str):
    sql = normalization(sql)
    sql = sql_nested_query_tmp_name_convert(sql)
    fks = build_fk_map_from_json(tables)
    pks = build_primary_key_list_from_json(tables)
    c2t = build_column_table_map_from_json(tables)
    schema = {}
    db_path = os.path.join(db_dir, db_name, db_name + ".sqlite")
    schema[db_name] = Schema(get_schema(db_path))
    toks = tokenize(sql)
    tables_with_alias = get_tables_with_alias(schema[db_name].schema, toks)
    sql = value_fix(sql, c2t, db_name, tables_with_alias, db_path)
    # if "group" in sql.lower():
    #     sql = groupby_fix(sql, db_name, pks[db_name])
    # if "join" in sql.lower():
    #     sql = join_fix(sql, fks[db_name], db_name, tables_with_alias)
    # if "order" in sql.lower():
    #     sql = orderby_fix(sql)
    sql = sql.replace("SSSSSSSSSSS", "")
    sql = sql.replace("SSSSSSSSSS", "")
    for agg in ["count", "min", "max", "sum", "avg"]:
        sql = sql.replace(f" {agg} ( ", f" {agg}( ")
    return normalization(sql)


if __name__ == '__main__':
    with open("./datasets/preds/picard_top8.txt", 'r') as f:
        preds = f.readlines()[::9]
    with open("./datasets/spider/dev.json", 'r') as f:
        dev = json.load(f)
    for i, (ins, pred) in enumerate(zip(dev, preds)):
        print(i, end="\t")
        try:
            pred = normalization(pred)
            sql = error_fix(ins['db_id'], "./datasets/spider/database", "./datasets/spider/tables.json", pred)
            sql = " ".join(sql.split())
            sql = normalization(sql)
            # sql = (sql + " ").replace(" ( ", "(").replace(" ) ", ") ").strip().replace(" , ", ", ")
            # for op in [">", "<", "=", " in"]:
            #     # sql = sql.replace(f"{op}(", f"{op} (")
            print(ins['query'])
            for agg in ["count", "min", "max", "sum", "avg"]:
                pred = pred.replace(f" {agg} ( ", f" {agg}( ")
            print(pred)
            print(sql == pred.strip(), sql)
        except Exception as e:
            print("[ERROR]")
            print(ins['db_id'], pred, e, sep="\t")
        print()
