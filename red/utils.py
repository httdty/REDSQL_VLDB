# -*- coding: utf-8 -*-
# @Time    : 2023/12/21 14:20
# @Author  :
# @Email   :  
# @File    : utils.py
# @Software: PyCharm
import asyncio
import os
from functools import lru_cache
from typing import Tuple, Any

from bug_fix.consistency import postprocess, remove_distinct, TIMEOUT, replace_cur_year, get_cursor_from_path


async def exec_on_db_(sqlite_path: str, query: str) -> Tuple[str, Any]:
    query = replace_cur_year(query)
    cursor = get_cursor_from_path(sqlite_path)
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        # column names
        column_names = [description[0] for description in cursor.description]
        cursor.close()
        cursor.connection.close()
        return "result", (result, column_names)
    except Exception as e:
        cursor.close()
        cursor.connection.close()
        return "exception", e


async def exec_on_db(
        sqlite_path: str, query: str, process_id: str = "", timeout: int = TIMEOUT
) -> Tuple[str, Any]:
    try:
        return await asyncio.wait_for(exec_on_db_(sqlite_path, query), timeout)
    except asyncio.TimeoutError:
        return ('exception', TimeoutError)
    except Exception as e:
        return ("exception", e)


@lru_cache(maxsize=64, typed=False)
def get_exec_output(
        db: str,
        sql: str,
        keep_distinct: bool = False,
):
    # post-process the prediction.
    # e.g. removing spaces between ">" and "="
    sql = postprocess(sql)

    if not keep_distinct:
        try:
            # if sqlparse can't parse p_str, we should not even try to execute it
            sql = remove_distinct(sql)
        except Exception as e:
            return "exception", []

    db_dir = os.path.dirname(db)
    db_paths = [os.path.join(db_dir, basename) for basename in os.listdir(db_dir) if ".sqlite" in basename]
    for db_path in db_paths:
        flag, sql_denotation = asyncio.run(exec_on_db(db_path, sql))
        return flag, sql_denotation