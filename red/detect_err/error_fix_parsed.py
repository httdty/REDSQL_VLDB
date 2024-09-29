from main.preprocessing import normalization
from red.detect_err.process_sql import get_sql_loc, Schema, get_schema, tokenize
import json
import os
import sqlite3


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


def del_dif(sql, idx_list, name):
    toks = tokenize(sql)
    a = idx_list[name][0]
    b = idx_list[name][1]
    for i in range(a, b):
        toks[i] = 'ERRORRRRRRRRRR'
    result = " ".join(toks)
    return result


def like_check(db_path, table, col, pattern):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    sql = f"SELECT COUNT(*) FROM {table} WHERE {col} like {pattern}"
    cursor.execute(sql)
    num = cursor.fetchall()
    if num[0][0] == 0:
        return False
    return True


def value_check(db_path, table, col, val):
    if isinstance(val, int) or isinstance(val, float) or str(val).isdigit():
        return True
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    sql = f"SELECT COUNT(*) FROM {table} WHERE {col} = {val}"
    cursor.execute(sql)
    num = cursor.fetchall()
    if num[0][0] == 0:
        return False
    return True


def get_sub_sql(sql, sid, eid):
    parts = tokenize(sql)
    temp = []
    for idx in range(sid, eid):
        if parts[idx] == '(':
            num = 1
            idx += 1
            while num:
                if parts[idx] == '(':
                    num += 1
                if parts[idx] == ')':
                    num -= 1
                if num:
                    temp.append(parts[idx])
                idx += 1
            break
    results = ' '.join(temp)
    return results


def sub_check(db_path, table, col, subsql):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    sql = f"SELECT COUNT(*) FROM {table} WHERE {col} = ({subsql})"
    cursor.execute(sql)
    num = cursor.fetchall()
    if num[0][0] == 0:
        return False
    return True


def value_fix(sql, where_clause, db_name, db_dir, db_path, tables, idx_list, fks, pks):
    del_not = False
    for item in where_clause:
        if del_not:
            break
        if item == "and" or item == "or":
            continue

        # between
        if item[1] == 1:
            val1 = item[3]
            val2 = item[4]
            temp = item[2][1][1]
            parts = temp.strip("_").split(".")
            table = parts[0]
            col = parts[1]
            if not (value_check(db_path, table, col, val1) and value_check(db_path, table, col, val2)):
                del_not = True
            continue
        # in,exists
        if item[1] == 8 or item[1] == 11:
            subsql = get_sub_sql(sql, idx_list['where'][0], idx_list['where'][1])
            new_subsql = parsed_sql_error_fix(db_name, db_dir, tables, subsql)
            sql.replace(subsql, new_subsql)
            continue

        # like
        if item[1] == 9:
            pattern = item[3]
            temp = item[2][1][1]
            parts = temp.strip("_").split(".")
            table = parts[0]
            col = parts[1]
            if not like_check(db_path, table, col, pattern):
                del_not = True
            continue

        # is,not暂时没处理

        # <,>,>=,<=
        if item[1] == 3 or item[1] == 4 or item[1] == 5 or item[1] == 6:
            continue

        # =,!=,<>
        if isinstance(item[3], dict):
            temp = item[2][1][1]
            parts = temp.strip("_").split(".")
            table = parts[0]
            col = parts[1]
            subsql = get_sub_sql(sql, idx_list['where'][0], idx_list['where'][1])
            if not sub_check(db_path, table, col, subsql):
                del_not = True
                continue
            new_subsql = parsed_sql_error_fix(db_name, db_dir, tables, subsql)
            sql = sql.replace(subsql, new_subsql)
            continue
        val = item[3]
        temp = item[2][1][1]
        parts = temp.strip("_").split(".")
        table = parts[0]
        col = parts[1]
        if not value_check(db_path, table, col, val):
            del_not = True
    if del_not:
        sql = del_dif(sql, idx_list, "where")
    return sql


def is_primary(name, pks):
    parts = name.strip("_").split(".")
    for col in pks:
        if col == parts[1]:
            return True
    return False


def get_table_units(clause):
    tables = []
    tables = clause['from']['table_units']
    results = []
    for item in tables:
        results.append(item[1])
    return results


def is_multitables(parsed_sql):
    if len(parsed_sql['from']['table_units']) > 1:
        return True
    tables = []
    table = parsed_sql['from']['table_units'][0][1]
    if parsed_sql['intersect']:
        tables = get_table_units(parsed_sql['intersect'])
        if len(tables) > 1:
            return True
        elif tables[0] != table:
            return True
    if parsed_sql['union']:
        tables = get_table_units(parsed_sql['union'])
        if len(tables) > 1:
            return True
        elif tables[0] != table:
            return True
    if parsed_sql['except']:
        tables = get_table_units(parsed_sql['except'])
        if len(tables) > 1:
            return True
        elif tables[0] != table:
            return True
    for item in parsed_sql['where']:
        if isinstance(item[3], dict):
            tables = get_table_units(item[3])
            if len(tables) > 1:
                return True
            elif tables[0] != table:
                return True
    return False


def groupby_fix(sql, parsed_sql, groupby_clause, idx_list, pks):
    if len(groupby_clause) > 1:
        return sql
    if is_multitables(parsed_sql):
        return sql
    if is_primary(groupby_clause[0][1], pks):
        sql = ""
    return sql


def join_fix(sql, join_clause, idx_list, fks):
    for item in join_clause:
        if item == 'and':
            continue
        col1 = item[2][1][1].strip("_")
        col2 = item[3][1].strip("_")
        if col1 not in fks or col2 not in fks:
            sql = del_dif(sql, idx_list, "from")
            return sql
        if fks[col1] == col2 or fks[col2] == col1:
            return sql
        else:
            sql = del_dif(sql, idx_list, "from")
            return sql


def orderby_fix(sql, select_clause, db_name, db_path, idx_list):
    if select_clause[1][0][0] == 0:
        return sql
    else:
        sql = del_dif(sql, idx_list, "select")
        sql = del_dif(sql, idx_list, "orderBy")
    return sql


def get_sec_sql(sql, sql_ops, sidx, eidx):
    parts = tokenize(sql)
    temp = []
    for i in range(sidx, eidx):
        temp.append(parts[i])
    result = " ".join(temp)
    return result


def error_fix(sql, parsed_sql, db_name, db_dir, db_path, tables, idx_list, fks, pks):
    if parsed_sql['where']:
        sql = value_fix(sql, parsed_sql['where'], db_name, db_dir, db_path, tables, idx_list, fks, pks)
    if parsed_sql['groupBy']:
        sql = groupby_fix(sql, parsed_sql, parsed_sql['groupBy'], idx_list, pks[db_name])
        if sql == "":
            return sql
    if parsed_sql['from']['conds']:
        sql = join_fix(sql, parsed_sql['from']['conds'], idx_list, fks[db_name])
    if parsed_sql['orderBy']:
        sql = orderby_fix(sql, parsed_sql['select'], db_name, db_path, idx_list)
    if parsed_sql['intersect']:
        subsql = get_sec_sql(sql, 'intersect', idx_list['intersect'][0], idx_list['intersect'][1])
        new_subsql = parsed_sql_error_fix(db_name, db_dir, tables, subsql)
        sql = sql.replace(subsql, new_subsql)
    if parsed_sql['union']:
        subsql = get_sec_sql(sql, 'union', idx_list['union'][0], idx_list['union'][1])
        new_subsql = parsed_sql_error_fix(db_name, db_dir, tables, subsql)
        sql = sql.replace(subsql, new_subsql)
    if parsed_sql['except']:
        subsql = get_sec_sql(sql, 'except', idx_list['except'][0], idx_list['except'][1])
        new_subsql = parsed_sql_error_fix(db_name, db_dir, tables, subsql)
        sql = sql.replace(subsql, new_subsql)
    return sql


def parsed_sql_error_fix(db_name: str, db_dir: str, tables: str, sql: str):
    sql = sql_nested_query_tmp_name_convert(sql)
    fks = build_fk_map_from_json(tables)
    pks = build_primary_key_list_from_json(tables)
    schema = {}
    db_path = os.path.join(db_dir, db_name, db_name + ".sqlite")
    schema[db_name] = Schema(get_schema(db_path))
    idx_list = {
        "except": [],
        "from": [],
        "groupBy": [],
        "having": [],
        "intersect": [],
        "limit": [],
        "orderBy": [],
        "select": [],
        "union": [],
        "where": [],
    }
    idx_list, parsed_sql = get_sql_loc(schema[db_name], sql, idx_list)
    # return parsed_sql
    sql = error_fix(sql, parsed_sql, db_name, db_dir, db_path, tables, idx_list, fks, pks)
    sql = sql.replace(" ERRORRRRRRRRRR", "").replace("SSSSSSSSSS", "").replace(" errorrrrrrrrrr", "")
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
            sql = parsed_sql_error_fix(ins['db_id'], "./datasets/spider/database", "./datasets/spider/tables.json",
                                       pred)
            sql = " ".join(sql.split())
            sql = normalization(sql)
            # sql = (sql + " ").replace(" ( ", "(").replace(" ) ", ") ").strip().replace(" , ", ", ")
            # for op in [">", "<", "=", " in"]:
            #     # sql = sql.replace(f"{op}(", f"{op} (")
            print(ins['query'])
            # for agg in ["count", "min", "max", "sum", "avg"]:
            #     pred = pred.replace(f" {agg} ( ", f" {agg}( ")
            print(pred)
            print(sql == pred.strip(), sql)
        except Exception as e:
            print("[ERROR]")
            print(ins['db_id'], pred, e, sep="\t")
        print()
