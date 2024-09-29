# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 11:09
# @Author  :
# @Email   :  
# @File    : unit.py
# @Software: PyCharm

import sqlite3
import threading
from functools import lru_cache
from typing import List
from multiprocessing import Pool, TimeoutError

from sql_metadata.token import SQLToken

threadLock = threading.Lock()
TIMEOUT = 30


def locate_subquery(sql_tokens: List[SQLToken]):
    mask = []
    in_sub_query = 0
    current_level = -1
    for i, token in enumerate(sql_tokens):
        if token.is_left_parenthesis and token.next_token_not_comment.value.upper() == "SELECT" and in_sub_query == 0:
            current_level = token.parenthesis_level
            in_sub_query = 1
        mask.append(in_sub_query)
        if token.is_right_parenthesis and token.parenthesis_level == current_level - 1:
            current_level = -1
            in_sub_query = 0
    return mask


def sql_tokens_to_sql(tokens):
    sql = ""
    all_sub = True
    level = min([token.parenthesis_level for token in tokens])
    for token in tokens:
        if not token.is_left_parenthesis and not token.is_right_parenthesis and token.parenthesis_level == level:
            all_sub = False
        if token.is_name and isinstance(token.value, str) and (
                " " in token.value or "-" in token.value) and not token.value.startswith("'"):
            if "." in token.value:
                tab, col = token.value.split(".", maxsplit=1)
                if " " in tab or "-" in tab:
                    tab = f"`{tab}`"
                if " " in col or "-" in col:
                    col = f"`{col}`"
                sql += f"{tab}.{col}"
            else:
                if " " in token.value or "-" in token.value:
                    sql += f"`{token.value}`"
                else:
                    sql += f"{token.value}"
        else:
            sql += f"{token.stringified_token}"
    sql = sql.strip()
    if all_sub:
        sql = sql[1:-1].strip()
    return sql


def handle_invalid_utf8(text):
    try:
        return text.decode('utf-8')
    except UnicodeDecodeError:
        return text.decode('utf-8', 'ignore')


# @lru_cache(maxsize=256)
def execute_sql(db_path, sql):
    """
    在子进程中执行SQL查询并返回结果。
    """
    # 注意：每个进程应该创建自己的数据库连接
    conn = sqlite3.connect(db_path)
    conn.text_factory = handle_invalid_utf8
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        result = cursor.description, cursor.fetchall()
    except Exception as e:
        result = 'exception', e
    finally:
        cursor.close()
        conn.close()
    return result


def _init_worker():
    """
    忽略 KeyboardInterrupt，在子进程中。
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def exec_sql_list(db_path, sql_list, num_worker=10, timeout=TIMEOUT):
    outputs = []
    with Pool(processes=num_worker, initializer=_init_worker) as pool:
        results = []

        for sql in sql_list:
            result = pool.apply_async(execute_sql, (db_path, sql))
            results.append((sql, result))

        for sql, result in results:
            try:
                output = result.get(timeout=timeout)
                # print(f"Results for '{sql}':", output)
            except TimeoutError:
                # print(f"Query timeout for: '{sql}'")
                output = 'exception', TimeoutError(f"Timeout with time budget {timeout}")
            except Exception as e:
                output = 'exception', e
            outputs.append(output)
    return outputs


def is_number(s):
    if not s:
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_int(s):
    if not s:
        return True
    try:
        return "." not in str(s) and int(s) == float(s)
    except ValueError:
        return False


def all_is_number(value_list: List):
    for value in value_list:
        if not is_number(value):
            return False
    return True


def all_is_int(value_list: List):
    for value in value_list:
        if not is_int(value):
            return False
    return True


def type_of_query(db_path, sql):
    _, types = execute_sql(db_path, sql)
    types = [t[0] for t in types]
    types = set(types)
    if "null" in types:
        types.remove("null")
    return types
