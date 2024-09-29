# -*- coding: utf-8 -*-
# @Time    : 2024/3/22 14:45
# @Author  :
# @Email   :  
# @File    : red_parser.py
# @Software: PyCharm

import json
import re
import time

from copy import deepcopy
from typing import List
from sql_metadata import Parser
from sql_metadata.token import SQLToken

from red.parser.schema import Schema
from red.parser.schema_item import Table, Column
from red.parser.utils import sql_tokens_to_sql, locate_subquery, is_number, execute_sql, all_is_int, all_is_number, \
    is_int
from red.parser.report import Report, BugLevel


def parse_as_unit(tokens: List[SQLToken], db_schema: Schema):
    if not tokens:
        return None
    if tokens[0].value.upper() == "DISTINCT":
        tokens = tokens[1:]

    all_sub = True
    level = min([token.parenthesis_level for token in tokens])
    for token in tokens:
        if not token.is_left_parenthesis and not token.is_right_parenthesis and token.parenthesis_level == level:
            all_sub = False
            break

    if len(tokens) == 1:
        col = db_schema.get_column(tokens[0].value)
        if col:
            return col
        else:
            return tokens[0].value
    elif tokens[0].is_left_parenthesis and tokens[0].next_token_not_comment.value.upper() == "SELECT" and all_sub:
        return Query(sql_tokens_to_sql(tokens), deepcopy(db_schema))
    else:
        return Expression(tokens, db_schema)


def op_data_type(op):
    if op is None:
        return None

    if isinstance(op, Expression):
        return op.data_type
    elif isinstance(op, JointPredicate):
        return "INTEGER"
    elif isinstance(op, Column):
        col_types = op.col_type.split("&")
        if "REAL" in col_types:
            return "REAL"
        elif "INTEGER" in col_types:
            return "INTEGER"
        else:
            return "TEXT"
    elif isinstance(op, Query):
        if "REAL" in op.exec_table.cols[0].col_type:
            return "REAL"
        elif "INTEGER" in op.exec_table.cols[0].col_type:
            return "INTEGER"
        else:
            return "TEXT"
    elif isinstance(op, str):
        op = op.strip("'")
        op = op.strip('"')
        if op.upper() == "NULL":
            return ""
        elif is_int(op):
            return "INTEGER"
        elif is_number(op):
            return "REAL"
        else:
            return "TEXT"
    else:
        raise TypeError("Do not support such data type")


class Predicate:
    CMP = {
        '>', '<', '>=', '<=', '=', '!=', '!=', 'IN', 'NOT IN', 'LIKE', 'NOT LIKE', 'BETWEEN', "IS"
    }

    _SEG = {"SELECT", "WHERE", "FROM", "HAVING", "GROUP BY", "ORDER BY", "LIMIT", "ON", ",", "AND", "OR",
            "WHEN", "THEN", "("}

    def __init__(self, cmp_token: SQLToken, db_schema: Schema):
        if cmp_token.value.upper() in self.CMP:
            self.CMP = cmp_token.value.upper()
        else:
            raise ValueError("Do not support such predicate")
        self.sub_query_list = []
        self.sql_tokens = [cmp_token]
        self.cmp_token = cmp_token
        self._cmp = self.cmp_token.value.upper()
        self._level = self.cmp_token.parenthesis_level
        self.db_schema = db_schema
        self.ops = []  # SubQuery, Column, Value, Expression
        self.parse()

    def parse(self):
        seg_list = []
        # Left parse
        token = self.cmp_token
        tmp_seg = []
        while True:
            token = token.previous_token_not_comment
            if token is None or token.position == -1:
                break
            elif token.parenthesis_level == self._level and token.value.upper() in self._SEG:
                break
            elif token.is_keyword and "JOIN" in token.value.upper():
                break
            elif token.parenthesis_level < self._level:
                break
            self.sql_tokens.insert(0, token)
            if token.parenthesis_level > self._level:
                tmp_seg.insert(0, token)
            elif not (token.value.upper() == "NOT" and
                      token.next_token_not_comment.value.upper() in {"BETWEEN", "LIKE", "IN"}):
                tmp_seg.insert(0, token)

        seg_list.append(tmp_seg)

        # Right parse
        tmp_seg = []
        token = self.cmp_token
        num = int(self._cmp == 'BETWEEN')
        while True:
            token = token.next_token_not_comment
            if token is None or token.position == -1:
                break
            elif token.parenthesis_level == self._level and token.value.upper() in self._SEG:
                if self._cmp == "BETWEEN" and token.value.upper() == "AND" and num > 0:
                    num -= 0
                    self.sql_tokens.append(token)
                    seg_list.append(tmp_seg)
                    tmp_seg = []
                    continue
                else:
                    break
            elif token.parenthesis_level == self._level and token.is_keyword and "JOIN" in token.value.upper():
                break
            elif token.parenthesis_level < self._level:
                break

            self.sql_tokens.append(token)
            tmp_seg.append(token)

        if tmp_seg:
            seg_list.append(tmp_seg)

        # Parse object
        for seg in seg_list:
            self.ops.append(parse_as_unit(seg, self.db_schema))
        return

    def __str__(self):
        return self.sql_str

    def validate(self):
        res = []

        # Error: Logical error for `all constant`
        all_constant = True
        for op in self.ops:
            if isinstance(op, Column):
                all_constant = False
            elif isinstance(op, Expression):
                res.extend(op.validate())
                all_constant = all_constant and op.is_constant()
            elif isinstance(op, Query):
                res.extend(op.validate())
                all_constant = all_constant and op.exec_table.is_single_value()
        if all_constant:
            desc = f"Predicate is comparing {len(self.ops)} constants;"
            res.append(Report(BugLevel.ERROR, self.sql_str, desc))

        # Error: Logical error for same element
        selected_columns = []
        same_element = False
        for op in self.ops:
            if isinstance(op, Column):
                if op in selected_columns:
                    same_element = True
                    break
                selected_columns.append(op)
        if same_element:
            desc = f"Predicate is comparing two same columns;"
            res.append(Report(BugLevel.ERROR, self.sql_str, desc))

        # Error: Subquery type not supported
        op = self.ops[-1]
        if isinstance(op, Query):
            if not op.exec_table.is_single_column():
                desc = f"The result column number of {str(op)} is not 1;"
                res.append(Report(BugLevel.ERROR, self.sql_str, desc))
            is_list = not op.exec_table.is_single_value()
            is_null = op.exec_table.is_null()
            if is_null and self._cmp not in {"IS", "IS NOT"}:
                desc = f"Result of {str(op)} is NULL, 'IS/IS NOT' comparison is needed;"
                res.append(Report(BugLevel.ERROR, self.sql_str, desc))
            elif is_list and self._cmp not in {"IN", "NOT IN"}:
                desc = f"Result of {str(op)} has more than one values, 'IN/NOT IN' comparison is needed;"
                res.append(Report(BugLevel.ERROR, self.sql_str, desc))

        op_types = []
        for op in self.ops:
            op_types.append(self.cmp_type(op))

        # Error: IN/NOT IN must be LIST as the second operand
        if self._cmp in {"IN", "NOT IN"} and isinstance(self.ops[-1], str):
            if not (self.ops[-1].strip().stratswith("(")
                    and self.ops[-1].strip().endswith(")")
                    and "," in self.ops[-1]):
                desc = f"The second operand of {self._cmp} must be a List;"
                res.append(Report(BugLevel.ERROR, self.sql_str, desc))

        # Error: Unsupported data type comparison
        elif self._cmp in {"LIKE", "NOT LIKE"}:
            if not (isinstance(self.ops[-1], str) and "%" in self.ops[-1]):
                # desc = f"The second operand of {self._cmp} need a wildcard %."
                desc = f"Do not need {self._cmp} for a const TEXT matching predicate;"
                res.append(Report(BugLevel.ERROR, self.sql_str, desc))
            sql = f"SELECT COUNT(*) FROM {self.ops[0].tab.tab_name} WHERE {self.ops[0].col_name} LIKE {self.ops[-1]}"
            _, num = execute_sql(self.db_schema.db_path, sql)
            if not num:
                desc = f"No value Like {self.ops[-1]} in column {self.ops[0].col_name}"
                res.append(Report(BugLevel.INFO, self.sql_str, desc))
        elif self._cmp == "IS":
            if not (isinstance(self.ops[-1], str) and "NULL" in self.ops[-1]):
                desc = f"The second operand of {self._cmp} need to be NULL or NOT NULL;"
                res.append(Report(BugLevel.ERROR, self.sql_str, desc))

        # INFO: No such value
        if self._cmp == "=":
            op1 = self.ops[0]
            op2 = self.ops[1]
            if isinstance(op1, Column) and isinstance(op2, str) and self.cmp_type(op1) == "TEXT" and op1.values:
                if op2.strip("'").strip('"') not in op1.values:
                    desc = f"No such value of {op2} in column {op1};"
                    res.append(Report(BugLevel.INFO, self.sql_str, desc))
            elif isinstance(op2, Column) and isinstance(op1, str) and self.cmp_type(op2) == "TEXT" and op2.values:
                if op1.strip("'").strip('"') not in op2.values:
                    desc = f"No such value of {op1} in column {op2};"
                    res.append(Report(BugLevel.INFO, self.sql_str, desc))
        return res

    def get_used_schema(self):
        tables = set()
        columns = set()
        for op in self.ops:
            if isinstance(op, str) or isinstance(op, int) or isinstance(op, float) or isinstance(op, SQLToken):
                continue
            elif isinstance(op, Column):
                columns.add(op.tab.tab_name + "." + op.col_name)
            elif isinstance(op, Table):
                tables.add(op.tab_name)
            else:
                tables_, columns_ = op.get_used_schema()
                tables.update(tables_)
                columns.update(columns_)
        return tables, columns

    def cmp_type(self, op):
        def _simple_type(op_type):
            if "INTEGER" in op_type or "REAL" in op_type:
                return "NUMERIC"
            else:
                return "TEXT"

        if isinstance(op, Column):
            if "INTEGER" in op.col_type or "REAL" in op.col_type:
                return "NUMERIC"
            else:
                return _simple_type(op.col_type)
        elif isinstance(op, Expression):
            if "INTEGER" in op.data_type or "REAL" in op.data_type:
                return "NUMERIC"
            else:
                return _simple_type(op.data_type)
        elif isinstance(op, Query):
            if op.exec_table.is_single_column():
                if "INTEGER" in op.exec_table.cols[0].col_type or "REAL" in op.exec_table.cols[0].col_type:
                    return "NUMERIC"
                else:
                    return _simple_type(op.exec_table.cols[0].col_type)
        else:
            op = op.strip("'")
            op = op.strip('"')
            if is_number(op):
                return "NUMERIC"
            elif op.startswith("(") and op.endswith(")"):
                values = op.strip()[1:-1].split(",")
                print("WARNING values = op.strip()[1:-1].split(", ") " * 3000)

                return _simple_type(f"{self.cmp_type(values[0])}")
            else:
                return "TEXT"

    @property
    def sql_str(self):
        return sql_tokens_to_sql(self.sql_tokens)


class JointPredicate:
    _CON = {"AND", "OR"}

    def __init__(self, sql_tokens: List[SQLToken], db_schema: Schema):
        self.sql_tokens = sql_tokens
        self.db_schema = db_schema
        self.subquery_mask = locate_subquery(self.sql_tokens)
        self.level = min([token.parenthesis_level for token in self.sql_tokens])
        self.predicates = []
        self.parse()
        self.ops = self.predicates

    def in_subquery(self, i):
        return self.subquery_mask[i] > 0

    def parse(self):
        tmp_tokens = []
        is_sub_joint_predicate = False
        for i, token in enumerate(self.sql_tokens):
            if self.in_subquery(i):
                continue
            if token.parenthesis_level > self.level:
                if i == 0 or (token.is_left_parenthesis and self._is_con(token.previous_token_not_comment)):
                    is_sub_joint_predicate = True
                tmp_tokens.append(token)
            elif token.parenthesis_level == self.level and token.is_right_parenthesis:
                if is_sub_joint_predicate:
                    self.predicates.append(JointPredicate(tmp_tokens[1:], self.db_schema))
                is_sub_joint_predicate = False
                tmp_tokens = []
            elif self._is_con(token):
                self.predicates.append(token.value.upper())
            elif token.value in Predicate.CMP:
                self.predicates.append(Predicate(token, self.db_schema))
        return

    def _is_con(self, token: SQLToken):
        if token.value.upper() not in self._CON:
            return False
        if token.value.upper() == "OR":
            return True
        elif token.value.upper() == "AND":
            p_tok = token
            while p_tok is not None:
                p_tok = p_tok.previous_token_not_comment
                if p_tok.parenthesis_level != token.parenthesis_level:
                    continue
                elif p_tok.is_keyword or p_tok.is_left_parenthesis:
                    return p_tok.value.upper() != "BETWEEN"
            return True
        return False

    def validate(self):
        res = []
        for predicate in self.predicates:
            if isinstance(predicate, str):
                continue
            res.extend(predicate.validate())

        # ERROR: Logical bug for Logical Mutually Exclusive
        # e.g. 'a < b and a > c', but b < c
        col_cond = {}
        and_list = [0 for _ in self.predicates]
        idx = 1
        for i, predicate in enumerate(self.predicates):
            and_list[i] = idx
            if predicate == "OR":
                idx += 1

        for i, predicate in enumerate(self.predicates):
            if isinstance(predicate, Predicate):
                for op in predicate.ops:
                    if isinstance(op, Column):
                        if op.tab.alias:
                            key = (op, list(op.tab.alias)[0])
                        else:
                            key = (op, 0)
                        if key in col_cond:
                            col_cond[key].append((predicate, and_list[i]))
                        else:
                            col_cond[key] = [(predicate, and_list[i])]

        for key, conds in col_cond.items():
            if isinstance(key[0], Column):
                if len(conds) == 1:
                    continue
                if "REAL" in key[0].col_type or "INTEGER" in key[0].col_type:
                    left = float('-inf')
                    right = float('inf')
                    equal_str = ""
                    not_equal = []
                    in_list = []
                    not_in_list = []
                    equal = float('inf')
                    idx = conds[0][1]
                    for cond in conds:
                        if cond[1] == idx:
                            if isinstance(cond[0], Predicate) and (
                                    isinstance(cond[0].ops[1], float) or isinstance(cond[0].ops[1], int)):
                                if (cond[0].CMP == "<" or cond[0].CMP == "<=") and float(cond[0].ops[1]) < right:
                                    right = float(cond[0].ops[1])
                                elif (cond[0].CMP[0] == ">" or cond[0].CMP == ">=") and float(
                                        cond[0].ops[1]) > left:
                                    left = float(cond[0].ops[1])
                                elif cond[0].CMP == "BETWEEN":
                                    if float(cond[0].ops[1]) > left:
                                        left = float(cond[0].ops[1])
                                    if float(cond[0].ops[2]) < right:
                                        right = float(cond[0].ops[2])
                                elif cond[0].CMP == "IN":
                                    for op in cond[0].ops[1].ops[0].ops:
                                        if op != ",":
                                            in_list.append(float(op))
                                elif cond[0].CMP == "NOT IN":
                                    for op in cond[0].ops[1].ops[0].ops:
                                        if op != ",":
                                            not_in_list.append(float(op))
                                elif cond[0].CMP == "!=":
                                    not_equal.append(float(cond[0].ops[1]))
                                elif cond[0].CMP == "=":
                                    if equal_str == "" and equal == float('inf'):
                                        equal = float(cond[0].ops[1])
                                    else:
                                        desc = f"The value of one cell can't equal to different value"
                                        res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                            elif isinstance(cond[0], Predicate) and isinstance(cond[0].ops[1], str):
                                if cond[0].CMP == "=":
                                    if equal_str == "" and equal == float('inf'):
                                        equal_str = cond[0].ops[1]
                                    else:
                                        desc = f"The value of one cell can't equal to different value"
                                        res.append(Report(BugLevel.ERROR, self.sql_str, desc))

                        else:
                            if left > right:
                                desc = f"Logical bug: The conditional intervals do not overlap."
                                res.append(Report(BugLevel.ERROR, self.sql_str, desc))

                            if equal != float('inf'):
                                if equal > right or equal < left:
                                    desc = f"Logical bug: The conditional intervals do not overlap."
                                    res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                                if not_equal:
                                    if equal in not_equal:
                                        desc = f"The value can't equal and not equal to the same value"
                                        res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                            if in_list:
                                flag = False
                                for num in in_list:
                                    if right > num > left:
                                        flag = True
                                        break
                                if not flag:
                                    desc = f"Logical bug: The conditional intervals do not overlap."
                                    res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                                if equal != float('inf') and equal not in in_list:
                                    desc = f"The equal value is not in the IN_list"
                                    res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                            if not_in_list:
                                if equal != float('inf') and equal in not_in_list:
                                    desc = f"The value can't equal and not equal to the same value"
                                    res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                            idx = cond[1]
                            left = float('-inf')
                            right = float('inf')
                            not_equal = []
                            in_list = []
                            not_in_list = []
                            equal = float('inf')
                            equal_str = ""

                elif "TEXT" in key[0].col_type:
                    equal = ""
                    left = ""
                    right = ""
                    not_equal = []
                    in_list = []
                    not_in_list = []
                    patterns = []
                    not_patterns = []
                    idx = conds[0][1]
                    for cond in conds:
                        if cond[1] == idx:
                            if isinstance(cond[0], Predicate) and isinstance(cond[0].ops[1], str):
                                if cond[0].CMP == "=":
                                    if equal == "":
                                        equal = cond[0].ops[1]
                                    else:
                                        desc = f"The value of one cell can't equal to different value"
                                        res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                                elif (cond[0].CMP == "<" or cond[0].CMP == "<=") and (
                                        right == "" or cond[0].ops[1] < right):
                                    right = cond[0].ops[1]
                                elif (cond[0].CMP == ">" or cond[0].CMP == ">=") and (
                                        left == "" or cond[0].ops[1] > left):
                                    left = cond[0].ops[1]
                                elif cond[0].CMP == "BETWEEN":
                                    if left == "" or cond[0].ops[1] > left:
                                        left = cond[0].ops[1]
                                    if right == "" or cond[0].ops[2] < right:
                                        right = cond[0].ops[2]
                                elif cond[0].CMP == "!=":
                                    not_equal.append(cond[0].ops[1])
                                elif cond[0].CMP == "IN":
                                    for op in cond[0].ops[1].ops[0].ops:
                                        if op != ",":
                                            in_list.append(op)
                                elif cond[0].CMP == "NOT IN":
                                    for op in cond[0].ops[1].ops[0].ops:
                                        if op != ",":
                                            not_in_list.append(op)
                                elif cond[0].CMP == "LIKE":
                                    patterns.append(cond[0].ops[1])
                                elif cond[0].CMP == "NOT LIKE":
                                    not_patterns.append(cond[0].ops[1])
                        else:
                            if right != "" and left != "":
                                if right < left:
                                    desc = f"Logical bug: The conditional intervals do not overlap."
                                    res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                                if equal != "" and (equal > right or equal < left):
                                    desc = f"Logical bug: The conditional intervals do not overlap."
                                    res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                            if equal != "" and not_equal:
                                if equal in not_equal:
                                    desc = f"The value can't equal and not equal to the same value"
                                    res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                            if equal != "" and patterns:
                                for pattern in patterns:
                                    pat = re.escape(pattern)
                                    pat = pat.replace('%', '.*').replace('_', '.')
                                    repat = re.compile(pat, re.IGNORECASE)
                                    if not repat.match(equal):
                                        desc = f"The value can't satisfy the need of LIKE"
                                        res.append(Report(BugLevel.INFO, self.sql_str, desc))
                            if equal != "" and not_patterns:
                                for pattern in not_patterns:
                                    pat = re.escape(pattern)
                                    pat = pat.replace('%', '.*').replace('_', '.')
                                    repat = re.compile(pat, re.IGNORECASE)
                                    if repat.match(equal):
                                        desc = f"The value can't satisfy the need of NOT LIKE"
                                        res.append(Report(BugLevel.INFO, self.sql_str, desc))
                            if in_list:
                                flag = False
                                for string in in_list:
                                    if right > string > left:
                                        flag = True
                                        break
                                if not flag:
                                    desc = f"Logical bug: The conditional intervals do not overlap."
                                    res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                                if equal != "" and equal not in in_list:
                                    desc = f"The equal value is not in the IN_list"
                                    res.append(Report(BugLevel.INFO, self.sql_str, desc))
                            if not_in_list:
                                if equal != "" and equal in not_in_list:
                                    desc = f"The value can't equal and not equal to the same value"
                                    res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                            equal = ""
                            left = ""
                            right = ""
                            not_equal = []
                            in_list = []
                            not_in_list = []
                            patterns = []
                            not_patterns = []
                            idx = cond[1]
        return res

    def get_used_schema(self):
        tables = set()
        columns = set()
        for op in self.ops:
            if isinstance(op, str) or isinstance(op, int) or isinstance(op, float) or isinstance(op, SQLToken):
                continue
            elif isinstance(op, Column):
                columns.add(op.tab.tab_name + "." + op.col_name)
            elif isinstance(op, Table):
                tables.add(op.tab_name)
            else:
                tables_, columns_ = op.get_used_schema()
                tables.update(tables_)
                columns.update(columns_)
        return tables, columns

    @property
    def sql_str(self):
        return sql_tokens_to_sql(self.sql_tokens)

    def __str__(self):
        return self.sql_str


class Expression:
    with open("./red/parser/knowledge_base/expression.json", 'r') as f:
        rules = json.load(f)

    _CAL = {"+", "-", "*", "/", "||"}
    _CMP = {'>', '<', '>=', '<=', '=', '!=', 'IN', 'NOT IN', 'LIKE', 'NOT LIKE', 'BETWEEN', "IS"}
    _TYPE_MAP = {
        ("REAL", "REAL"): "REAL",
        ("REAL", "INTEGER"): "REAL",
        ("INTEGER", "REAL"): "REAL",
        ("INTEGER", "INTEGER"): "INTEGER",
        ("REAL", ""): "REAL",
        ("", "REAL"): "REAL",
        ("INTEGER", ""): "INTEGER",
        ("", "INTEGER"): "INTEGER",
    }

    def __init__(self, sql_tokens: List[SQLToken], db_schema: Schema, exp_type: str = ""):
        self.sql_tokens = sql_tokens
        self.db_schema = db_schema
        self.exp_type = exp_type  # CASE, IIF, CAST, FUN, CAL, LIST
        self.ops = []
        self.column = None
        self.data_type = "TEXT"
        self.level = min([token.parenthesis_level for token in self.sql_tokens])
        self.subquery_mask = locate_subquery(self.sql_tokens)
        self.parse()

    def in_subquery(self, i):
        return self.subquery_mask[i] > 0

    def _detect_exp_type(self):
        if self.sql_tokens[0].value.upper() == "CASE" and self.sql_tokens[-1].value.upper() == "END":
            case_list = [token.parenthesis_level == self.level and token.value.upper() == "CASE" for token in
                         self.sql_tokens[1:-1]]
            if not any(case_list):
                self.exp_type = "CASE"
        elif self.sql_tokens[0].value.upper() == "CAST":
            cast_list = [token.parenthesis_level == self.level for token in self.sql_tokens[1:-1]]
            if not any(cast_list):
                self.exp_type = "CAST"
                self.sql_tokens = self.sql_tokens[2:-1]
        # Detect type
        if not self.exp_type:
            for i, token in enumerate(self.sql_tokens):
                if (token.parenthesis_level == self.level and
                        (token.value in self._CAL or token.value.upper() in self._CMP)):
                    self.exp_type = "CAL"
            if not self.exp_type:
                if self.sql_tokens[0].is_left_parenthesis:
                    self.exp_type = "CAL"
                elif self.sql_tokens[0].value.upper() == "IIF":
                    self.exp_type = "IIF"
                else:
                    self.exp_type = "FUN"  # IIF, CAST, FUN

    def parse(self):
        self._detect_exp_type()

        if self.exp_type == "CAL":  # Calculation parse
            tmp_tokens = []
            exp_type = "CAL"
            is_predicate = False
            for i, token in enumerate(self.sql_tokens):
                if token.parenthesis_level == self.level and token.value.upper() in self._CMP:
                    is_predicate = True
                    break
            if is_predicate:
                self.ops.append(JointPredicate(self.sql_tokens, self.db_schema))
            else:
                for i, token in enumerate(self.sql_tokens):
                    # Begin (
                    if token.parenthesis_level > self.level:
                        tmp_tokens.append(token)
                    elif token.is_right_parenthesis:
                        tmp_tokens.append(token)
                        if exp_type == "FUN":
                            self.ops.append(Expression(tmp_tokens, self.db_schema, exp_type))
                        elif len(tmp_tokens) > 2 and tmp_tokens[1].value.upper() == "SELECT":
                            self.ops.append(Query(sql_tokens_to_sql(tmp_tokens[1:-1]), deepcopy(self.db_schema)))
                        else:
                            self.ops.append(Expression(tmp_tokens[1:-1], self.db_schema, exp_type))
                        tmp_tokens = []
                        exp_type = "CAL"
                    elif token.value.upper() == "CAST":
                        exp_type = "CAST"
                    elif token.value.upper() == "IIF":
                        exp_type = "IIF"
                    elif self._is_function(token):
                        exp_type = "FUN"
                        tmp_tokens.append(token)
                    elif token.value.upper() in self._CAL or token.value.upper() in self._CMP:
                        self.ops.append(token)
                    else:
                        self.ops.append(parse_as_unit([token], self.db_schema))

        elif self.exp_type == "CASE":  # CASE parse
            # Sep into clauses
            cond_tokens = []
            then_tokens = []
            else_tokens = []
            current = cond_tokens
            for i, token in enumerate(self.sql_tokens):
                if token.value.upper() == "CASE" and token.parenthesis_level == self.level:
                    current = cond_tokens
                elif token.value.upper() == "WHEN" and token.parenthesis_level == self.level:
                    if then_tokens:
                        self.ops.append(parse_as_unit(then_tokens, self.db_schema))
                        current.clear()
                    current = cond_tokens
                elif token.value.upper() == "THEN" and token.parenthesis_level == self.level:
                    # Parse `conds`
                    self.ops.append(JointPredicate(cond_tokens, self.db_schema))
                    current.clear()
                    current = then_tokens
                elif token.value.upper() == "ELSE" and token.parenthesis_level == self.level:
                    # Parse `THEN`
                    self.ops.append(parse_as_unit(current, self.db_schema))
                    current.clear()
                    current = else_tokens
                elif token.value.upper() == "END" and token.parenthesis_level == self.level:
                    # Parse `THEN` or `ELSE`
                    self.ops.append(parse_as_unit(current, self.db_schema))
                    current.clear()
                    current = None
                else:
                    current.append(token)
            pass
            # self.ops.append(JointPredicate(cond_tokens, self.db_schema))
            # self.ops.append(parse_as_unit(then_tokens, self.db_schema))

        elif self.exp_type == "IIF":  # IIF parse
            # Sep into clauses
            cond_tokens = []
            first_tokens = []
            second_tokens = []
            current = cond_tokens
            for token in self.sql_tokens[2:-1]:
                if token.value.upper() == "," and token.parenthesis_level == self.level + 1:
                    if current is cond_tokens:
                        current = first_tokens
                    elif current is first_tokens:
                        current = second_tokens
                else:
                    current.append(token)
            # Parse `conds`
            self.ops.append(JointPredicate(cond_tokens, self.db_schema))
            # Parse `first result`
            self.ops.append(parse_as_unit(first_tokens, self.db_schema))
            # Parse `second result`
            self.ops.append(parse_as_unit(second_tokens, self.db_schema))

        elif self.exp_type == "CAST":  # CAST parse
            target_type = self.sql_tokens[-1].value.upper()
            if "FLOAT" in target_type:
                target_type = "REAL"
            if "INT" in target_type:
                target_type = "INTEGER"
            self.ops.append(parse_as_unit(self.sql_tokens[:-2], self.db_schema))
            self.ops.append(target_type)

        elif self.exp_type == "FUN":  # Function parse
            # Sep into clauses
            self.ops.append(self.sql_tokens[0].value.upper())
            parameters = [[]]
            for i, token in enumerate(self.sql_tokens[2:-1]):
                if token.value.upper() == "," and token.parenthesis_level == self.level + 1:
                    parameters.append([])
                else:
                    parameters[-1].append(token)
                # Parse `parameters`
            for parameter in parameters:
                if parameter:
                    self.ops.append(parse_as_unit(parameter, self.db_schema))

    def validate(self):
        res = []
        for op in self.ops:
            if isinstance(op, SQLToken) or isinstance(op, Column) or isinstance(op, str):
                continue
            elif isinstance(op, Query) or isinstance(op, Expression) or isinstance(op, JointPredicate):
                res.extend(op.validate())

        if self.exp_type == "CAL":  # Calculation validation
            if isinstance(self.ops[0], SQLToken):
                self.data_type = op_data_type(self.ops[1])
            else:
                self.data_type = op_data_type(self.ops[0])
            if all([isinstance(op, str)] for op in self.ops) and all([op == "," for op in self.ops[1::2]]):
                for op in self.ops[0::2]:
                    right_type = op_data_type(op)
                    if right_type == "REAL" and self.data_type != "TEXT":
                        self.data_type = "REAL"
                    elif right_type == "INTEGER" and self.data_type != "TEXT":
                        continue
                    elif right_type == "TEXT":
                        self.data_type = "TEXT"
            else:
                for i, op in enumerate(self.ops):
                    if i == 0 or isinstance(op, SQLToken):
                        continue
                    elif isinstance(op, str) and op.upper() in {"AND", "OR"}:
                        continue
                    else:
                        right_type = op_data_type(op)  # right op
                        if isinstance(self.ops[i - 1], SQLToken) and self.ops[i - 1].value.upper() in self._CMP:
                            self.data_type = "INTEGER"
                        elif right_type == "REAL" and self.data_type != "TEXT":
                            self.data_type = "REAL"
                        elif right_type == self.data_type == "INTEGER" and isinstance(self.ops[i - 1], SQLToken) and \
                                self.ops[i - 1].value == "/":
                            desc = (f"Dividing two integers results in loss of precision, "
                                    f"'* 1.0' or 'CAST AS REAL' might be helpful;"
                                    f"\n[REFINE SUGGESTION]: Use '*1.0' or 'CAST AS REAL' to turn the integer to float")
                            res.append(Report(BugLevel.WARNING, self.sql_str, desc))
                        elif right_type == "INTEGER" and self.data_type != "TEXT":
                            continue
                        elif self.data_type == right_type and right_type == "TEXT":
                            if isinstance(self.ops[i - 1], SQLToken):
                                cal = self.ops[i - 1].value.upper()
                            else:
                                cal = str(self.ops[i - 1])
                            if cal in self._CAL and cal != "||":
                                desc = f"Can not operate the {cal} between two TEXT;"
                                res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                        else:
                            cal = self.ops[i - 1].value.upper()
                            desc = (f"Can not operate the {cal} between {self.data_type}({self.ops[i - 2]})"
                                    f" and {right_type}({self.ops[i]});")
                            res.append(Report(BugLevel.ERROR, self.sql_str, desc))

        elif self.exp_type == "CASE" or self.exp_type == "IIF":  # CASE/IIF validation
            left_type = op_data_type(self.ops[1])
            for op in self.ops[2:]:
                if not isinstance(op, JointPredicate):
                    right_type = op_data_type(op)
                    if not right_type:
                        right_type = left_type
                    if not self.data_type:
                        self.data_type = right_type
                    elif "TEXT" in (left_type, right_type):
                        self.data_type = "TEXT"
                    else:
                        self.data_type = self._TYPE_MAP[(left_type, right_type)]
            if not self.data_type:
                self.data_type = left_type

        elif self.exp_type == "CAST":  # CAST parse
            op0_data_type = op_data_type(self.ops[0])
            if op0_data_type == "TEXT":
                desc = f"Can not CAST a TEXT({self.ops[0]}) to a {self.ops[-1]};"
                res.append(Report(BugLevel.ERROR, self.sql_str, desc))
            elif op0_data_type == "REAL" and self.ops[-1].upper() == "INTEGER":
                desc = f"Convert a REAL({self.ops[0]}) to a {self.ops[-1]} might results in loss of precision;"
                res.append(Report(BugLevel.WARNING, self.sql_str, desc))
            self.data_type = self.ops[-1]

        elif self.exp_type == "FUN":  # Function parse
            # Sep into clauses
            fun_name = self.ops[0]
            if fun_name not in self.rules:
                desc = f"Function {fun_name} not supported by SQLite;"
                res.append(Report(BugLevel.ERROR, self.sql_str, desc))
            else:
                rule = self.rules[fun_name]
                allowed_num = 0
                for allowed_type in rule['input']:
                    if "NULL" in allowed_type:
                        break
                    allowed_num += 1

                if len(self.ops[1:]) < allowed_num or len(self.ops[1:]) > len(rule['input']):
                    desc = f"Function {fun_name} does not support {len(self.ops[1:])} parameters as input;"
                    res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                else:
                    output_type = "TEXT"
                    for i, (parameter, allowed_type) in enumerate(zip(self.ops[1:], rule['input'][:len(self.ops[1:])])):
                        allowed_type = allowed_type.split("|")
                        parameter_type = op_data_type(parameter)
                        if parameter_type in allowed_type:
                            idx = allowed_type.index(parameter_type)
                            if "|" in rule['output'] and len(allowed_type) == len(rule['output'].split("|")):
                                output_type = rule['output'].split("|")[idx]
                            elif "|" in rule['output']:
                                continue
                            else:
                                output_type = rule['output']
                        else:
                            desc = (
                                f"The type of {parameter} is {parameter_type}, but function {i + 1}-th parameter of "
                                f"{fun_name} should be {allowed_type};")
                            res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                    if output_type == "TEXT":
                        # further detect
                        # print("Need to detect the real data type")
                        has_column = False
                        for op in self.ops[1:]:
                            col = get_col(op)
                            if col:
                                has_column = True
                                tab_name = f"`{col.tab.tab_name}`"
                                for alias in col.tab.alias:
                                    if f"{alias}.".upper() in self.sql_str.upper():
                                        tab_name = f"`{col.tab.tab_name}` AS {alias}"

                                sql = (f"SELECT {self.sql_str} FROM {tab_name} "
                                       f"WHERE `{col.col_name}` IS NOT NULL LIMIT 1000;")
                                header, values = execute_sql(self.db_schema.db_path, sql)
                                if header != 'exception':
                                    values = [value[0] for value in values]
                                    if all_is_int(values):
                                        output_type = "INTEGER"
                                    elif all_is_number(values):
                                        output_type = "REAL"
                                else:
                                    output_type = "TEXT"
                        if not has_column:
                            sql = f"SELECT {self.sql_str};"
                            _, values = execute_sql(self.db_schema.db_path, sql)
                            values = [value[0] for value in values]
                            if all_is_int(values):
                                output_type = "INTEGER"
                            elif all_is_number(values):
                                output_type = "REAL"
                    self.data_type = output_type

        return res

    def get_used_schema(self):
        tables = set()
        columns = set()
        for op in self.ops:
            if isinstance(op, str) or isinstance(op, int) or isinstance(op, float) or isinstance(op, SQLToken):
                continue
            elif isinstance(op, Column):
                columns.add(op.tab.tab_name + "." + op.col_name)
            elif isinstance(op, Table):
                tables.add(op.tab_name)
            else:
                tables_, columns_ = op.get_used_schema()
                tables.update(tables_)
                columns.update(columns_)
        return tables, columns

    def is_constant(self):
        all_constant = True
        for op in self.ops:
            if isinstance(op, Column):
                all_constant = all_constant and op.is_single_value()
            elif isinstance(op, Expression):
                all_constant = all_constant and op.is_constant()
            elif isinstance(op, Query):
                all_constant = all_constant and op.exec_table.is_single_value()
            elif isinstance(op, str) and op == "*" and self.exp_type == "FUN":
                all_constant = False
        return False

    def _is_function(self, token: SQLToken) -> bool:
        return token.value.upper() in self.rules and token.next_token_not_comment.is_left_parenthesis

    def __str__(self):
        return self.sql_str

    @property
    def sql_str(self):
        return sql_tokens_to_sql(self.sql_tokens)


def get_col(unit):
    if isinstance(unit, Column):
        return unit
    elif isinstance(unit, Query):
        return unit.exec_table.cols[0]
    elif isinstance(unit, Expression):
        for op in unit.ops:
            col = get_col(op)
            if col:
                return col
    return None


class Clause:
    def __init__(self, sql_tokens: List[SQLToken], db_schema: Schema, **clauses):
        self.sql_tokens = sql_tokens
        self.db_schema = db_schema
        self.clauses = clauses
        self.sql_str = sql_tokens_to_sql(self.sql_tokens)
        self.subquery_mask = locate_subquery(self.sql_tokens)
        self.ops = []
        self.parse()

    def in_subquery(self, i):
        return self.subquery_mask[i] > 0

    def _parse_seg(self):
        sql_token_seg = []
        tmp_sql_tokens = []
        select_level = self.sql_tokens[0].parenthesis_level
        for i, token in enumerate(self.sql_tokens):
            if i == 0:
                continue
            if token.value == "," and token.parenthesis_level == select_level:
                sql_token_seg.append(tmp_sql_tokens)
                tmp_sql_tokens = []
            else:
                tmp_sql_tokens.append(token)
        if tmp_sql_tokens:
            sql_token_seg.append(tmp_sql_tokens)
        return sql_token_seg

    def parse(self):
        pass

    def validate(self):
        pass

    def get_used_schema(self):
        tables = set()
        columns = set()
        for op in self.ops:
            if isinstance(op, str) or isinstance(op, int) or isinstance(op, float) or isinstance(op, SQLToken):
                continue
            elif isinstance(op, Column):
                columns.add(op.tab.tab_name + "." + op.col_name)
            elif isinstance(op, Table):
                tables.add(op.tab_name)
            else:
                tables_, columns_ = op.get_used_schema()
                tables.update(tables_)
                columns.update(columns_)
        return tables, columns

    def __str__(self):
        return sql_tokens_to_sql(self.sql_tokens)


class FromClause(Clause):
    def __init__(self, sql_tokens: List[SQLToken], db_schema: Schema, **clauses):
        self.join_conds: List[JointPredicate] = []
        self.sub_queries = []

        super().__init__(sql_tokens, db_schema, **clauses)

    def parse(self):
        is_table = True
        sub_query = []
        join_con_tokens = []
        for i, token in enumerate(self.sql_tokens):
            if self.in_subquery(i):
                sub_query.append(token)
                continue
            if sub_query:
                self.sub_queries.append(Query(sql_tokens_to_sql(sub_query), deepcopy(self.db_schema)))
                self.db_schema.add_query_table(self.sub_queries[-1])
                sub_query = []
            if token.value.upper() == "ON":
                is_table = False
            elif token.is_keyword and ("JOIN" in token.value.upper() or "FROM" == token.value.upper()):
                if join_con_tokens:
                    self.join_conds.append(JointPredicate(join_con_tokens, self.db_schema))
                    join_con_tokens = []
                is_table = True
            elif not is_table:
                join_con_tokens.append(token)
            elif (token.is_alias_definition or
                  token.is_alias_of_table_or_alias_of_subquery and token.value.upper() != "AS"):
                self.db_schema.add_alias(token.value, self.db_schema.query_tabs[-1])
            elif token.is_name or self.db_schema.is_schema(token.value):
                table = self.db_schema.get_all_table(token.value)
                if is_table and table is not None:
                    self.db_schema.add_query_table(table)
        if sub_query:
            self.sub_queries.append(Query(sql_tokens_to_sql(sub_query), deepcopy(self.db_schema)))
            self.db_schema.add_query_table(self.sub_queries[-1])
        if join_con_tokens:
            self.join_conds.append(JointPredicate(join_con_tokens, self.db_schema))

        self.ops = self.join_conds + self.sub_queries + self.db_schema.query_tabs
        return

    def validate(self):
        res = []
        # Sub queries validation
        for query in self.sub_queries:
            res.extend(query.validate())

        for joint_cond in self.join_conds:
            res.extend(joint_cond.validate())
            for cond in joint_cond.predicates:
                if isinstance(cond, JointPredicate):
                    desc = f"Too complicate join condition;"
                    res.append(Report(BugLevel.WARNING, self.sql_str, desc))
                elif isinstance(cond, Predicate):
                    # ERROR: Join operands must be two Columns/Expressions
                    if not (isinstance(cond.ops[0], Column) or isinstance(cond.ops[0], Expression)):
                        desc = f"Operand {cond.ops[0]} is not a column;"
                        res.append(Report(BugLevel.WARNING, self.sql_str, desc))
                    if not (isinstance(cond.ops[1], Column) or isinstance(cond.ops[1], Expression)):
                        desc = f"Operand {cond.ops[1]} is not a column;"
                        res.append(Report(BugLevel.WARNING, self.sql_str, desc))

                    # ERROR: Only support equal join
                    if cond.cmp_token.value != "=":
                        desc = f"Join by {cond.cmp_token.value} is not supported;"
                        res.append(Report(BugLevel.WARNING, self.sql_str, desc))

                    # WARNING: No explicit PK-FK relation
                    op0 = cond.ops[0]
                    op1 = cond.ops[1]
                    if isinstance(op0, Column) and isinstance(op1, Column):
                        if not (op0.is_fk_with(op1) or op0 is op1):
                            tables = []
                            for op in self.ops:
                                if isinstance(op, Table):
                                    tables.append(op.tab_name)
                            path = self.db_schema.graph.find_minimum_spanning_tree(tables)
                            if path:
                                desc = (f"No explicitly defined PK-FK relationship between {op0} and {op1};"
                                        f"\n[REFINE SUGGESTION]: The suggested join path is {path}")
                            else:
                                desc = f"No explicitly defined PK-FK relationship between {op0} and {op1};"
                            res.append(Report(BugLevel.WARNING, self.sql_str, desc))

                            # ERROR: Can not be joined by the two columns
                            type0 = set(op0.col_type.split("&"))
                            type1 = set(op1.col_type.split("&"))
                            same = type0 & type1
                            if same:
                                value0 = set(op0.values)
                                value1 = set(op1.values)
                                common_values = value0 & value1
                                # TODO: could be better?
                                if not common_values:
                                    desc = f"{op0} and {op1} don't have the common value, can't be joined by these two columns;"
                                    res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                            else:
                                desc = f"{op0} and {op1} have different column types, can't be joined by these two columns;"
                                res.append(Report(BugLevel.ERROR, self.sql_str, desc))

        return res


class WhereClause(Clause):
    def __init__(self, sql_tokens: List[SQLToken], db_schema: Schema, **clauses):
        self.predicates = None

        super().__init__(sql_tokens, db_schema, **clauses)

    def parse(self):
        # Parse where predicates
        self.predicates = JointPredicate(self.sql_tokens[1:], self.db_schema)
        self.ops = [self.predicates]
        return

    def validate(self):
        res = []
        res.extend(self.predicates.validate())
        return res


class GroupbyClause(Clause):
    def __init__(self, sql_tokens: List[SQLToken], db_schema: Schema, **clauses):
        self.group_cols = []

        super().__init__(sql_tokens, db_schema, **clauses)

    def parse(self):
        # Parse group by cols
        sql_token_seg = self._parse_seg()
        for seg in sql_token_seg:
            self.group_cols.append(parse_as_unit(seg, self.db_schema))
        self.ops = self.group_cols
        return

    def validate(self):
        res = []
        for op in self.group_cols:
            if isinstance(op, Expression) or isinstance(op, Query):
                res.extend(op.validate())
            # WARNING: Group by PK but querying only one tab
            if isinstance(op, Column):
                if op.is_pk() and len(self.group_cols) == 1 and len(self.db_schema.query_tabs) == 1:
                    desc = f"Group by primary key when there is only one table is meaningless;"
                    res.append(Report(BugLevel.WARNING, self.sql_str, desc))
            # ERROR: Project non-aggregated columns
            for pro_op in self.clauses['SELECT'].projections:
                if isinstance(op, Column) and op not in self.group_cols:
                    desc = f"The selected column {pro_op} is not in the groupby clause and without aggregation;"
                    res.append(Report(BugLevel.ERROR, self.sql_str, desc))
        return res


class HavingClause(Clause):
    def __init__(self, sql_tokens: List[SQLToken], db_schema: Schema, **clauses):
        self.predicates = None

        super().__init__(sql_tokens, db_schema, **clauses)

    def parse(self):
        # Parse having predicates
        self.predicates = JointPredicate(self.sql_tokens[1:], self.db_schema)
        self.ops = [self.predicates]
        return

    def validate(self):
        res = []
        res.extend(self.predicates.validate())
        for op in self.predicates.ops:
            if isinstance(op, Column):
                # ERROR: Having clause include non-aggregated columns
                if op not in self.clauses['GROUP BY'].group_cols:
                    desc = f"The column {op} in the HAVING clause is not in the groupby clause and without aggregation"
                    res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                    # TODO: ...
        return res


class OrderbyClause(Clause):
    def __init__(self, sql_tokens: List[SQLToken], db_schema: Schema, **clauses):
        self.order_cols = []
        self.direction = []

        super().__init__(sql_tokens, db_schema, **clauses)

    def parse(self):
        sql_token_seg = self._parse_seg()
        for seg in sql_token_seg:
            # Parse order direction
            if seg[-1].value.upper() in {"DESC", "ASC"}:
                self.direction.append(seg[-1].value.upper())
                seg = seg[:-1]
            else:
                self.direction.append("ASC")
            # Parse order cols
            self.order_cols.append(parse_as_unit(seg, self.db_schema))
        self.ops = self.order_cols + self.direction
        return

    def validate(self):
        res = []
        for op in self.order_cols:
            if isinstance(op, Query):
                res.extend(op.validate())
            # WARNING: Order by has_null col(Cross-Clause)
            if isinstance(op, Column):
                sql = ""
                for instance in self.clauses:
                    if instance == "SELECT":
                        continue
                    if instance == "WHERE":
                        sql = sql + f" WHERE ({self.clauses['WHERE'].sql_str.split('WHERE')[1]}) and `{op.col_name}` IS NULL"
                    else:
                        sql = sql + self.clauses[instance].sql_str
                sql = "SELECT COUNT(*) " + sql + ";"
                flag, num = execute_sql(self.db_schema.db_path, sql)
                if flag != "exception" and num[0][0] and self.direction[0] == "ASC":
                    desc = (f"The column in `{op.col_name}` has null value and order direction is ASC, "
                            f"which will make the value of the top results is NULL, "
                            f"`NOT NULL` predicate for `{op.col_name}` might be helpful.")
                    res.append(Report(BugLevel.WARNING, self.sql_str, desc))
            if isinstance(op, Expression):
                sql = ""
                for instance in self.clauses:
                    if instance == "SELECT":
                        continue
                    if instance == "WHERE":
                        sql = sql + f" WHERE ({self.clauses['WHERE'].sql_str.split('WHERE')[1]}) and `{op.sql_str}` IS NULL"
                    else:
                        sql = sql + self.clauses[instance].sql_str
                sql = "SELECT COUNT(*) " + sql + ";"
                flag, num = execute_sql(self.db_schema.db_path, sql)
                if flag != "exception" and num[0][0] and self.direction[0] == "ASC":
                    desc = (f"The column in `{op.sql_str}` has null value and order direction is ASC, "
                            f"which will make the value of the top results is NULL, "
                            f"`NOT NULL` predicate for `{op.sql_str}` might be helpful.")
                    res.append(Report(BugLevel.WARNING, self.sql_str, desc))
        return res


class LimitClause(Clause):
    def __init__(self, sql_tokens: List[SQLToken], db_schema: Schema, **clauses):
        self.num = -1

        super().__init__(sql_tokens, db_schema, **clauses)

    def parse(self):
        val = self.sql_tokens[-1].value
        if val.isnumeric():
            self.num = int(val)
        self.ops = []
        return

    def validate(self):
        res = []
        # TODO: no order by
        if 'ORDER BY' not in self.clauses:
            desc = f"Limit without ORDER BY is meaningless."
            res.append(Report(BugLevel.WARNING, self.sql_str, desc))
        return res


class SelectClause(Clause):
    def __init__(self, sql_tokens: List[SQLToken], db_schema: Schema, **clauses):
        self.projections = []

        super().__init__(sql_tokens, db_schema, **clauses)

    def parse(self):
        sql_token_seg = self._parse_seg()
        for seg in sql_token_seg:
            # Parse alias
            alias = None
            if (seg[-1].is_alias_definition and not seg[-1].is_right_parenthesis
                    and not seg[-1].value.upper() == "END" and not is_int(seg[-1].value)):
                alias = seg[-1].value
                if seg[-2].is_as_keyword:
                    seg = seg[:-2]
                else:
                    seg = seg[:-1]

            # Parse projections
            self.projections.append(parse_as_unit(seg, self.db_schema))
            if alias is not None:
                self.db_schema.add_alias(alias, self.projections[-1])

        self.ops = self.projections

    def validate(self):
        res = []
        for op in self.projections:
            if isinstance(op, Expression) or isinstance(op, Query):
                res.extend(op.validate())
        # TODO: 如果对做了Groupby，选了一个非groupby并且没有agg的值，这个SQL就是错的
        return res


class IUEClause(Clause):
    def __init__(self, sql_tokens: List[SQLToken], db_schema: Schema, **clauses):
        self.iue_keyword = ""
        self.left_query = None
        self.right_query = None
        super().__init__(sql_tokens, db_schema, **clauses)

    def parse(self):
        for i, token in enumerate(self.sql_tokens):
            # Locate IUE keyword
            if token.parenthesis_level == 0 and token.value.upper() in {
                'INTERSECT', 'INTERSECT ALL', 'UNION', 'UNION ALL', 'EXCEPT', 'EXCEPT ALL'
            }:
                left_sql = sql_tokens_to_sql(self.sql_tokens[:i])
                right_sql = sql_tokens_to_sql(self.sql_tokens[i + 1:])
                self.iue_keyword = f"{token.value.upper()}"
                self.left_query = Query(left_sql, self.db_schema)
                self.right_query = Query(right_sql, self.db_schema)
                self.ops = [self.left_query, self.right_query]
                break

    def validate(self):
        res = []
        res.extend(self.left_query.validate())
        res.extend(self.right_query.validate())

        # IUE validate
        r_num = len(self.right_query.clauses['SELECT'].projections)
        l_num = len(self.left_query.clauses['SELECT'].projections)
        # ERROR: IUE do not have same projection column num
        if r_num != l_num:
            desc = f"The right subquery and the left subquery don't select the same number of columns"
            res.append(Report(BugLevel.ERROR, self.sql_str, desc))
        # ERROR: IUE do not have same projection column types
        else:
            r_ops = self.right_query.clauses['SELECT'].projections
            l_ops = self.left_query.clauses['SELECT'].projections
            for i in range(r_num):
                if isinstance(r_ops[i], str):
                    if isinstance(l_ops[i], str):
                        continue
                    else:
                        desc = f"The right subquery and the left subquery select the columns of different types"
                        res.append(Report(BugLevel.ERROR, self.sql_str, desc))
                if isinstance(r_ops[i], Column):
                    r_types = set(r_ops[i].col_type.split("&"))
                else:
                    r_types = set(r_ops[i].data_type.split("&"))
                if isinstance(l_ops[i], Column):
                    l_types = set(l_ops[i].col_type.split("&"))
                else:
                    l_types = set(l_ops[i].data_type.split("&"))
                same = r_types & l_types
                if not same:
                    # TODO: why same projection, not same type?
                    desc = f"The right subquery and the left subquery select the columns of different types"
                    res.append(Report(BugLevel.ERROR, self.sql_str, desc))
        return res


class Query:
    _clause_dict = {
        'FROM': FromClause,
        'WHERE': WhereClause,
        'GROUP BY': GroupbyClause,
        'HAVING': HavingClause,
        'ORDER BY': OrderbyClause,
        'LIMIT': LimitClause,
        'SELECT': SelectClause,
        'INTERSECT': IUEClause,
        'INTERSECT ALL': IUEClause,
        'UNION': IUEClause,
        'UNION ALL': IUEClause,
        'EXCEPT': IUEClause,
        'EXCEPT ALL': IUEClause
    }
    _parse_order = [
        'INTERSECT', 'INTERSECT ALL', 'UNION', 'UNION ALL', 'EXCEPT', 'EXCEPT ALL',
        'FROM', 'SELECT', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT',
    ]
    _process_order = [
        'INTERSECT', 'INTERSECT ALL', 'UNION', 'UNION ALL', 'EXCEPT', 'EXCEPT ALL',
        'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'SELECT',
    ]

    def __init__(self, sql, db_schema=None):
        self.sql = sql.strip().strip(";")
        self._pre_process_sql()
        self.db_schema = db_schema
        self.clauses_tokens = dict()
        self.clauses = dict()
        self.table = None
        self.parser()

    def _pre_process_sql(self):
        sql = self.sql
        p = Parser(self.sql)
        for i, token in enumerate(p.tokens):
            if token.parenthesis_level == 0 and token.value.upper() in {
                'INTERSECT', 'INTERSECT ALL', 'UNION', 'UNION ALL', 'EXCEPT', 'EXCEPT ALL'
            }:
                left_sql = "(" + sql_tokens_to_sql(p.tokens[:i]) + ")"
                right_sql = sql_tokens_to_sql(p.tokens[i + 1:])
                if not right_sql.startswith("("):
                    right_sql = "(" + right_sql + ")"
                sql = left_sql + f" {token.value.upper()} " + right_sql
                return sql
        return sql

    def parser(self):
        p = Parser(self._pre_process_sql())
        current_clause = ""
        current_tokens = []

        # Sep clauses
        for token in p.tokens:
            t = token.value.upper()
            if t in self._clause_dict and current_clause != t and token.parenthesis_level == 0:
                # End Clause
                if current_clause:
                    self.clauses_tokens[current_clause] = current_tokens
                    current_tokens = []
                # New Clause Begin
                current_clause = t
            current_tokens.append(token)
        self.clauses_tokens[current_clause] = current_tokens

        # Parse clauses
        for current_clause in self._parse_order:
            if current_clause in self.clauses_tokens:
                clause_type = self._clause_dict[current_clause]
                self.clauses[current_clause] = clause_type(self.clauses_tokens[current_clause],
                                                           self.db_schema, **self.clauses)
                pass

    @property
    def exec_table(self) -> Table:
        if self.table:
            return self.table

        if "LIMIT" not in self.sql and "SELECT" in self.clauses:
            sql = self.sql + " LIMIT 10000"
        else:
            sql = self.sql
        db_path = self.db_schema.db_path
        headers, res = execute_sql(db_path, sql)
        if headers == "exception" and "no such column" in res.args[0]:
            raise ValueError(f"Do not support such kind of SQL {sql}")
        elif headers == "exception":
            print(self.sql, res)
            # input("Check?")
            raise ValueError(f"Do not support such kind of SQL {sql}")
        col_names = []
        col_types = []

        for i, projection in enumerate(headers):
            col_names.append(projection[0])
            values = [value[i] for value in res if value[i] is not None]
            if all_is_int(values):
                col_types.append("INTEGER")
            elif all_is_number(values):
                col_types.append("REAL")
            else:
                col_types.append("TEXT")
        self.table = Table(f"TMP{str(time.time())[8:]}".replace(".", ""), col_names, [], col_types)
        for i, col in enumerate(self.table.cols):
            values = [value[i] for value in res]
            col.set_values(values)
            col.has_null = None in values
        return self.table

    def validate(self):
        res = []
        for clause in self._parse_order:
            if clause in self.clauses:
                res.extend(self.clauses[clause].validate())
        is_null_table = True
        for col in self.exec_table.cols:
            if col.values:
                is_null_table = False
                break
        if is_null_table:
            if "(select max(" in self.sql.lower():
                pattern = r'\(select max\((\w+)\)'
                match = re.search(pattern, self.sql.lower())
                temp_col = match.group(1)
                desc = (f"The execution result of SQL is null. "
                        f"\n[REFINE SUGGESTION]: Use \"order by {temp_col} desc limit 1\" instead of subquery."
                        f" Don't use orderby and subquery together!")
                res.append(Report(BugLevel.INFO, self.sql, desc))
            elif "(select min(" in self.sql.lower():
                pattern = r'\(select min\((\w+)\)'
                match = re.search(pattern, self.sql.lower())
                temp_col = match.group(1)
                desc = (f"The execution result of SQL is null. "
                        f"\n[REFINE SUGGESTION]: Use \"order by {temp_col} asc limit 1\" instead of subquery."
                        f" Don't use orderby and subquery together!")
                res.append(Report(BugLevel.INFO, self.sql, desc))
            else:
                desc = (f"The execution result of SQL is null. "
                        f"\n[REFINE SUGGESTION]: No tuple satisfy all the predicate, wrong column might be included.")
                res.append(Report(BugLevel.INFO, self.sql, desc))
        else:
            if len(self.exec_table.cols) == 1:
                if len(set(self.exec_table.cols[0].values)) == 1 and len(self.exec_table.cols[0].values) > 1:
                    desc = (f"There is only single column and single value in the execution results."
                            f"\n[REFINE SUGGESTION]: Use \"DISTINCT\" in the select clause")
                    res.append(Report(BugLevel.INFO, self.sql, desc))
                if 'SELECT' in self.clauses:
                    if isinstance(self.clauses['SELECT'].projections[0], Expression):
                        if self.clauses['SELECT'].projections[0].exp_type == "FUN":
                            if "COUNT" in self.clauses['SELECT'].projections[0].sql_str and (
                                    float(self.exec_table.cols[0].values[0]) == 0):
                                desc = (f"The execution result of COUNT() of SQL is zero. "
                                        f"\n[REFINE SUGGESTION]: No tuple satisfy all the predicate, wrong column might be included.")
                                res.append(Report(BugLevel.INFO, self.sql, desc))

            # if "join" in self.sql.lower() and self.sql.lower().count('select') > 1:
            #     desc = f"Maybe The result of subquery is not in the join table, which makes the result of Query None"
            #     res.append(Report(BugLevel.INFO, self.sql, desc))

        # for bug in res:
        #     print(bug)
        return res

    def get_used_schema(self):
        tables = set()
        columns = set()
        for clause in self._parse_order:
            if clause in self.clauses:
                tables_, columns_ = self.clauses[clause].get_used_schema()
                tables.update(tables_)
                columns.update(columns_)
        return tables, columns

    def __str__(self):
        return self.sql
