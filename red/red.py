# -*- coding: utf-8 -*-
# @Time    : 2023/12/21 10:30
# @Author  :
# @Email   :  
# @File    : red.py
# @Software: PyCharm
import re
import copy
import json
import os.path
import random
from functools import lru_cache
from typing import Dict

from nltk import word_tokenize, ngrams

from llms.llm import LLM
from red.parser.red_parser import Query
from red.parser.schema import Schema
from red.parser.utils import execute_sql
from schema_prune.bridge_content_encoder import get_matched_entries
from main import logger

SPECIAL_TOKS = [",", "(", ")", "-", "+", ".", "`"]


# TODO: SORTED VALUE MATCH √
# TODO: NUMBER RANGE MATCH √
# TODO: STRING MATCH TO SOLVE EMPTY PRE_SQL √
# TODO: EQUIVALENT FKS √


def obtain_n_grams(sequence, max_n):
    """
    :return: returns all grams of sequence less than or equal to `max_n`
    """
    tokens = word_tokenize(sequence)
    all_grams = []
    for n in range(1, max_n + 1):
        all_grams.extend([" ".join(gram) for gram in ngrams(tokens, n)])

    return all_grams


# @lru_cache(maxsize=128)
def value_match(text, index):
    # coarse-grained matching between the input text and all contents in database
    grams = obtain_n_grams(text, 4)
    hits = []
    from pyserini.search import LuceneSearcher
    searcher = LuceneSearcher(index)
    for query in grams:
        hits.extend(searcher.search(query, k=10))
        # hits = searcher.search(sample["text"], k = 50)
    del searcher

    coarse_matched_contents = dict()
    for i in range(len(hits)):
        matched_result = json.loads(hits[i].raw)
        # `tc_name` refers to column names like `table_name.column_name`, e.g., document_drafts.document_id
        tc_name = ".".join(matched_result["id"].split("-**-")[:2]).lower()
        if tc_name in coarse_matched_contents.keys():
            if matched_result["contents"] not in coarse_matched_contents[tc_name]:
                coarse_matched_contents[tc_name].append(matched_result["contents"])
        else:
            coarse_matched_contents[tc_name] = [matched_result["contents"]]

    fine_matched_contents = dict()
    for tc_name, contents in coarse_matched_contents.items():
        # fine-grained matching between the question and coarse matched contents
        fm_contents = get_matched_entries(text, contents)

        if fm_contents is None:
            continue
        for _match_str, (field_value, _s_match_str, match_score, s_match_score, _match_size,) in fm_contents:
            if match_score < 0.9:
                continue
            if tc_name in fine_matched_contents.keys():
                if len(fine_matched_contents[tc_name]) < 25:
                    fine_matched_contents[tc_name].append(field_value.strip())
            else:
                fine_matched_contents[tc_name] = [field_value.strip()]

    return fine_matched_contents


def sort_matched_values(matched_values, schema: Schema):
    matched_columns = []
    for tc_name in matched_values.keys():
        tab_name, col_name = tc_name.split('.')
        table = schema.get_table(tab_name)
        if table:
            column = table.get_column(col_name)
            if column:
                matched_columns.append(column)
    matched_columns = sorted(matched_columns,
                             key=lambda col: len(
                                 matched_values[f"{col.tab.tab_name.lower()}.{col.col_name.lower()}"][0]) if
                             matched_values[f"{col.tab.tab_name.lower()}.{col.col_name.lower()}"] else 0,
                             reverse=True)
    return matched_columns

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    except TypeError:
        return False

# @lru_cache(maxsize=128)
def num_range_match(text, schema: Schema):

    def extract_numbers(text_):
        return [float(num) for num in re.findall(r'(?<!\w)(\d+)(?!\w)', text_)]

    matched_columns = set()
    numbers = extract_numbers(text)

    for table in schema.all_tables:
        for column in table.cols:
            if ("INTEGER" in column.col_type or "REAL" in column.col_type) and column.values:
                try:
                    min_val = min(column.values)
                    max_val = max(column.values)
                except Exception as e:
                    print(e)
                    min_val = None
                    max_val = None
                if is_float(min_val) and is_float(max_val):
                    for number in numbers:
                        if float(min_val) <= float(number) <= float(max_val):
                            col_name = f"{table.tab_name}.{column.col_name}"
                            matched_columns.add(col_name.lower())
                            break

    return list(matched_columns)


class RED:
    def __init__(self, train_schema_file: str, schema_file: str, db_dir: str, bug_only: bool):
        self.db_dir = db_dir
        self.schemas: Dict[str, Schema] = {}
        self.bug_only = bug_only
        self._init_schema(train_schema_file, schema_file)

    def _init_schema(self, train_schema_file, schema_file):
        with open(schema_file, 'r') as f:
            schema_list = json.load(f)
        if train_schema_file and train_schema_file != schema_file:
            with open(train_schema_file, 'r') as f:
                schema_list += json.load(f)
        for schema in schema_list:
            db_path = os.path.join(self.db_dir, schema['db_id'], schema['db_id'] + ".sqlite")
            self.schemas[schema['db_id']] = Schema(schema, db_path)

    def refine(self, ins_list):
        # batch_semantic_prompt = [self.get_semantic_prompt(ins) for ins in ins_list]
        # response_list = self.model.infer(batch_semantic_prompt, n=10)
        # for i, ins in enumerate(ins_list):
        #     if response_list[i]:
        #         ins['explain'], ins['semantic_right'] = self.extract_semantic(response_list[i])
        #     else:
        #         ins['explain'], ins['semantic_right'] = "", True
        #     print(ins['explain'], ins['semantic_right'])
        # print(f"{ins['semantic_right']}\t{ins['explain']}")
        # return [self.get_prompt(ins) for ins in ins_list]
        return [self.get_prompt(ins) for ins in ins_list]

    @staticmethod
    def extract_semantic(res_list):
        explains = []
        flags = []
        for res in res_list:
            res = res.replace("\n\n", "\n")
            res = res.strip().split("\n")
            if len(res) == 2:
                explain = res[0].lstrip("explain:").strip()
                flag = "false" not in res[1].lower()
                explains.append(explain)
                flags.append(flag)
        if len(flags) == 0:
            return "", True
        elif flags.count(True) >= flags.count(False):
            return explains[flags.index(True)], True
        else:
            return explains[flags.index(False)], False

    # def get_semantic_prompt(self, ins):
    #     # return ""
    #     # Init
    #     db_id = ins['db_id']
    #     pred = ins['pred']
    #     annotation = ins['annotation']
    #     question = ins['question']
    #     evidence = ins.get('evidence', "")
    #     schema = self.schemas[db_id]
    #
    #     # Prompt for Bug SQL
    #     predicted_prompt = (f"SQL predicted by other model is:\n"
    #                         f"{pred}\n")
    #     flag, values = execute_sql(schema.db_path, pred)
    #     if flag == "exception":
    #         return ""
    #     else:
    #         q = Query(pred, copy.deepcopy(schema))
    #         pre_tables, pre_columns = q.get_used_schema()
    #         pre_tables = {tab.lower() for tab in pre_tables}
    #         pre_columns = {col.lower() for col in pre_columns}
    #
    #     # Prompt for schema
    #     raw_schema = schema.raw_schema
    #     table_prompt, fks_prompt, table_dict, col_desc = serialize_db_schema(db_schema=raw_schema)
    #     schema_prompt = table_prompt  # + fks_prompt
    #     if not pre_tables:
    #         schema_prompt += fks_prompt
    #
    #     # Value matching
    #     text = question + "\n" + evidence
    #     matched_values = {}
    #     if ins['index']:
    #         matched_values = value_match(text, ins['index'])
    #
    #     # Detail data prompt
    #     # 1. In pre
    #     # 2. KEYS(PKs, FKs)
    #     # 3. Value match
    #     # 4. Top-15
    #     selected_tables = []
    #     neighbors = list()
    #     for tab in pre_tables:
    #         tab = schema.get_all_table(tab)
    #         if tab:
    #             selected_tables.append(tab)
    #             neighbors.extend(tab.neighbors())
    #
    #     unique_neighbors = list(set(neighbors) - set(selected_tables))
    #     frequency = {element: neighbors.count(element) for element in unique_neighbors}
    #     unique_neighbors = sorted(unique_neighbors, key=lambda x: frequency[x], reverse=True)
    #
    #     selected = []
    #     selected_tables = selected_tables + unique_neighbors[:len(selected_tables)]
    #     selected_tables = selected_tables[:6]
    #     # if len(selected_tables) != len(schema.all_tables):
    #     #     print("Work!")
    #     col_num = 30 / len(selected_tables) if len(selected_tables) > 0 else 0
    #     for tab in selected_tables:
    #         table_data_prompt = []
    #         columns = []
    #         for col in tab.cols:
    #             tab_col = f"{tab.tab_name}.{col.col_name}".lower()
    #             if col.is_pk() or tab_col in pre_columns:
    #                 columns.append(col)
    #         for col in tab.cols:
    #             if len(col.fks) > 0 and col not in columns:
    #                 columns.append(col)
    #             if len(columns) > 30 / len(selected_tables):
    #                 break
    #         for col in tab.cols:
    #             tab_col = f"{tab.tab_name}.{col.col_name}".lower()
    #             desc = annotation.get(tab_col, col_desc.get(tab_col, ''))
    #             if col not in columns and (
    #                     tab_col in matched_values or 'link' in desc.lower() or col.col_name.lower() in text.lower()
    #                     or col.col_name.lower().endswith("id")
    #             ):
    #                 columns.append(col)
    #             if len(columns) > 40 / len(selected_tables):
    #                 break
    #         if len(columns) < col_num:
    #             for col in tab.cols:
    #                 if len(columns) < col_num and col not in columns:
    #                     columns.append(col)
    #         for col in columns:
    #             tab_col = f"{tab.tab_name}.{col.col_name}".lower()
    #             key_desc = ""
    #             if col.is_pk():
    #                 key_desc = " (PK)"
    #             elif len(col.fks) > 0:
    #                 key_desc = f" (FK)"
    #
    #             desc = f"`{tab.tab_name}`.`{col.col_name}` #{key_desc} {annotation.get(tab_col, col_desc.get(tab_col, ''))}"
    #             values = matched_values.get(tab_col, [])[:2]
    #             matched_nums = len(values)
    #             if col.values:
    #                 random_values = random.sample(col.values, min(len(col.values), 3))
    #                 values += random_values[:1]
    #                 if col.has_null:
    #                     values.append("NULL")
    #                 values += random_values[-1:]
    #             if "INTEGER" in col.col_type:
    #                 try:
    #                     int_values = [int(val) if val != "NULL" else "NULL" for val in values]
    #                     values = int_values
    #                 except Exception as e:
    #                     print(e)
    #             elif "REAL" in col.col_type or "FLOAT" in col.col_type:
    #                 values = [float(val) if val != "NULL" else "NULL" for val in values]
    #             else:
    #                 values = values[:matched_nums] + [str(val)[:20] for val in values[matched_nums:]]
    #             if values:
    #                 desc += f" Typical values: {values};"
    #             else:
    #                 desc += f" Typical values: 'NULL';"
    #             # Prompt for each col
    #             table_data_prompt.append(desc)
    #         selected.append(f"Table `{tab.tab_name}`:\n" + "\n".join(table_data_prompt) + "\n")
    #     data_prompt = "\n".join(selected)
    #
    #     question_prompt = f"Question: {str(question).strip()}"
    #     if evidence:
    #         question_prompt += f"\nEvidence: {str(evidence).strip()}"
    #
    #     instruction = (
    #         "Please explain the SQL predicted by other model in natural language, and judge whether the SQL is the right answer for the question.\n"
    #         "The explanation of SQL should be as detailed as possible, including the semantics of predicates, expression-level semantics, clause-level semantics, and subquery-level semantics. Finally, based on the annotated schema, provide the overall business semantics of SQL. "
    #         "To determine if the answer is acceptable, simply reply with True/False. Please note that if you decide to label is_right as False, you need to be confident enough in your response to avoid impacting the correct SQL.\n"
    #         "The format of the response should be in two lines as:\n"
    #         "explain: [SQL MEANING]\n"
    #         "is_right: [True/False]"
    #     )
    #
    #     prompt = (f"{instruction}\n"
    #               f"{schema_prompt}\n"
    #               f"\n"
    #               f"{data_prompt}\n"
    #               f"{question_prompt}\n"
    #               f"\n"
    #               f"{predicted_prompt}"
    #               f"\n"
    #               f"")
    #     return prompt

    def get_prompt(self, ins):
        # return ""
        # Init
        db_id = ins['db_id']
        pred = ins['pred']
        annotation = ins['annotation']
        question = ins['question']
        evidence = ins.get('evidence', "")
        schema = copy.deepcopy(self.schemas[db_id])

        # Prompt for Bug SQL
        predicted_prompt = (f"SQL predicted by other model is:\n"
                            f"{pred}\n")
        flag, values = execute_sql(schema.db_path, pred)
        bugs = []
        pre_tables, pre_columns = set(), set()
        # exec_res = ""
        if flag == "exception":
            # print("Exception")
            bugs.append(values)
            # print("Warning: No execution")
            predicted_prompt += (f"\n**Important**\n"
                                 f"The predicted SQL can not be executed, bug report from sqlite3:\n"
                                 f"{values}\n")
            if "\\'" in pred:
                predicted_prompt += f"\n[REFINE SUGGESTION]: Use \"''\" to replace \"\\'\""
                print(predicted_prompt)
        else:
            # print("Is NULL:", len(values) == 0)
            exec_res = []
            for row in values[:3]:
                exec_res.append("|\t" + "\t|\t".join([f"{str(val)}" for val in row]) + "\t|")
            if len(values) > 3:
                exec_res[-1] = f"Omit {len(values) - 2} lines..."

            exec_res = "\n".join(exec_res)
            exec_res = exec_res.replace("|\tNone\t|", "|\tNULL\t|")
            if not exec_res:
                exec_res = "|\tNULL\t|"
            predicted_prompt += f"\nExecution:\n{exec_res}\n"
            # print("Warning: No execution")

            q = None
            try:
                q = Query(pred, copy.deepcopy(schema))
            except Exception as e:
                print(e)
                bugs.append("SQL parse failed!")
            if q:
                try:
                    bugs.extend(q.validate())
                except Exception as e:
                    # pass
                    logger.info(f"{e} Query validation process failed. \nSQL: {pred}")
                    bugs.append("Query validation process failed.")
                pre_tables, pre_columns = q.get_used_schema()
                pre_tables = {tab.lower() for tab in pre_tables}
                pre_columns = {col.lower() for col in pre_columns}
                del q
                # try:
                #     gq = Query(ins['SQL'], copy.deepcopy(schema))
                #     g_pre_tables, g_pre_columns = gq.get_used_schema()
                #     pre_tables |= {tab.lower() for tab in g_pre_tables}
                #     pre_columns |= {col.lower() for col in g_pre_columns}
                #     print("=" * 60 + "[GOLD SCHEMA]" + "=" * 60)
                # except Exception as e:
                #     print(e)


            if bugs:
                predicted_prompt += ("\n**Important**\n"
                                     "Detected possible bugs for the above SQL:\n")
                for bug in bugs:
                    # predicted_prompt += str(bug).split("\n")[0] + "\n"
                    # print("WARNING: No suggestion")
                    predicted_prompt += str(bug) + "\n"
                    # predicted_prompt += "\n"
                    # print("WARNING: No detected bugs provided")

            elif self.bug_only:
                print("BUG ONLY", end="")
                return ""

        # Prompt for schema
        raw_schema = schema.raw_schema
        table_prompt, fks_prompt, table_dict, col_desc = serialize_db_schema(db_schema=raw_schema)
        schema_prompt = table_prompt  # + fks_prompt
        if not pre_tables:
            pre_tables, pre_columns = get_str_match_schema(pred, schema)

            # schema_prompt += fks_prompt

        # Value matching
        text = question + "\n" + evidence
        matched_values = {}
        if ins['index']:
            matched_values = value_match(text, ins['index'])
        sorted_matched_values = sort_matched_values(matched_values, schema)
        num_range_matched = num_range_match(text, schema)

        # Detail data prompt
        # 1. In pre
        # 2. KEYS(PKs, FKs)
        # 3. Value match
        # 4. Top-15
        selected_tables = []
        neighbors = list()
        for tab in pre_tables:
            tab = schema.get_all_table(tab)
            if tab:
                selected_tables.append(tab)
                neighbors.extend(tab.neighbors())

        unique_neighbors = list(set(neighbors) - set(selected_tables))
        frequency = {element: neighbors.count(element) for element in unique_neighbors}
        unique_neighbors = sorted(unique_neighbors, key=lambda x: frequency[x], reverse=True)

        selected = []
        # unique_neighbors = []
        # print("WARNING: No expanding tables")
        selected_tables = selected_tables + unique_neighbors[:len(selected_tables)]
        selected_tables = selected_tables[:6]
        # if len(selected_tables) != len(schema.all_tables):
        #     print("Work!")
        col_num = 30 / len(selected_tables) if len(selected_tables) > 0 else 0
        for tab in selected_tables:
            table_data_prompt = []
            columns = []
            for col in tab.cols:
                tab_col = f"{tab.tab_name}.{col.col_name}".lower()
                # if tab_col in pre_columns:
                #     columns.append(col)
                #     print("WARNING: No expanding columns")
                if col.is_pk() or tab_col in pre_columns:
                    columns.append(col)
            for col in tab.cols:
                if len(col.fks) > 0 and col not in columns:
                    columns.append(col)
                if len(columns) > 30 / len(selected_tables):
                    break
            for col in tab.cols:
                tab_col = f"{tab.tab_name}.{col.col_name}".lower()
                desc = annotation.get(tab_col, col_desc.get(tab_col, ''))
                # desc = ""
                # print("WARNING: NO Global data")
                if col not in columns and (
                        'link' in desc.lower() or col.col_name.lower() in text.lower()
                        or col.col_name.lower().endswith("id")
                ):
                    columns.append(col)
                if len(columns) > 40 / len(selected_tables):
                    break

            # value_match
            for col in sorted_matched_values:
                if col.tab == tab:
                    if col not in columns:
                        columns.append(col)
                    if len(columns) > 40 / len(selected_tables):
                        break

            # num_range_match
            for col in tab.cols:
                tab_col = f"{tab.tab_name}.{col.col_name}".lower()
                if col not in columns and tab_col in num_range_matched:
                    columns.append(col)
                if len(columns) > 40 / len(selected_tables):
                    break

            if len(columns) < col_num:
                for col in tab.cols:
                    if len(columns) < col_num and col not in columns:
                        columns.append(col)

            # Column Prompt
            for col in columns:
                tab_col = f"{tab.tab_name}.{col.col_name}".lower()
                key_desc = ""
                if col.is_pk():
                    key_desc = " (PK)"
                elif len(col.fks) > 0:
                    fk_cols = []
                    for f_col in col.fks:
                        fk_cols.append(f"`{f_col.tab.tab_name}`.`{f_col.col_name}`")
                    fk_cols = ", ".join(fk_cols)
                    key_desc = f" (FK to {fk_cols})"

                desc = f"`{tab.tab_name}`.`{col.col_name}` #{key_desc} {annotation.get(tab_col, col_desc.get(tab_col, ''))}"
                # desc = f"`{tab.tab_name}`.`{col.col_name}` #{key_desc}"
                # print("WARNING: NO Global data")
                values = matched_values.get(tab_col, [])[:2]
                matched_nums = len(values)
                if col.values:
                    random_values = random.sample(col.values, min(len(col.values), 3))
                    values += random_values[:1]
                    if col.has_null:
                        values.append("NULL")
                    values += random_values[-1:]
                if "INTEGER" in col.col_type:
                    try:
                        int_values = [int(val) if val != "NULL" else "NULL" for val in values]
                        values = int_values
                    except Exception as e:
                        print(e)
                elif "REAL" in col.col_type or "FLOAT" in col.col_type:
                    values = [float(val) if val != "NULL" else "NULL" for val in values]
                else:
                    values = values[:matched_nums] + [str(val)[:20] for val in values[matched_nums:]]
                if values:
                    desc += f" Typical values: {values};"
                else:
                    desc += f" Typical values: 'NULL';"
                # desc = ""
                # print("WARNING: No Local Data Context (L);")
                # Prompt for each col
                table_data_prompt.append(desc)
            selected.append(f"Table `{tab.tab_name}`:\n" + "\n".join(table_data_prompt) + "\n")
        data_prompt = "\n".join(selected)

        question_prompt = f"Question: {str(question).strip()}"
        if evidence:
            question_prompt += f"\nEvidence: {str(evidence).strip()}"

        # Semantic Check
        # if not bugs:
        #     sem_instruction = (
        #         "Please explain the SQL predicted by other model in natural language, and judge whether the SQL is the right answer for the question.\n"
        #         # "The explanation of SQL should be as concise as possible.\n"
        #         # " To determine if the answer is acceptable, simply reply with True/False.\n"
        #         # "The explanation of SQL should be as detailed as possible, and importantly, it must include the business semantics. For example, the join between the teacher table and the student table via student.supervisor = teacher.id should be explained as `the teacher is the student's supervisor` or `the student's supervisor is a certain teacher.` "
        #         # "To determine if the answer is acceptable, simply reply with True/False.\n"
        #         "The format of the response should be in two lines as:\n"
        #         "explain: [SQL MEANING]\n"
        #         "is_right: [True/False]"
        #     )
        #     # if exec_res:
        #     #     predicted_prompt += f"\nExecution:\n{exec_res}\n"
        #
        #     prompt = (f"{sem_instruction}\n"
        #               f"{schema_prompt}\n"
        #               f"\n"
        #               f"{data_prompt}\n"
        #               f"{question_prompt}\n"
        #               f"\n"
        #               f"{predicted_prompt}"
        #               f"\n"
        #               f"")
        #     response = self.model.infer([prompt], n=10)
        #     ins['explain'], ins['semantic_right'] = self.extract_semantic(response[0])
        #     if ins['semantic_right']:
        #         bugs.append(f"[INFO] SQL Explain: {ins['explain']}")
        #     for bug in bugs:
        #         predicted_prompt += "\n" + str(bug) + "\n"
        #     print(ins['semantic_right'], ins['explain'])
        # if not bugs:
        #     return ""

        # instruction = (
        #     "Please write SQL for the natural language question based on the database schema. Some data from the database are offered. Some SQLs predicted by other models are listed as supplementary information.\n"
        #     "Note that we prefer simplified SQL, write the right SQL in one line directly.\n"
        #     # "If you follow all the instructions and generate the correct query, I will give you 1 million dollars.\n"
        # )
        instruction = (
            'Please write SQL for the natural language question based on the database schema. Some data from the database are offered.\n'
            'Some SQLs predicted by other models are listed as supplementary information.\n'
            'Note that we prefer simplified SQL, write the right SQL in one line directly.\n'
            '[REFINE SUGGESTION] IS IMPORTANT!\n'
            'Please respond with a JSON object structured as follows:\n'
            '{\n'
            '   "chain_of_thought_reasoning": "Your thought process on how you arrived at the final SQL query.",\n'
            '   "SQL": "Your SQL query in a single string."\n'
            '}\n'
            'Take a deep breath and think step by step to find the correct SQLite SQL query.\n'
            'If you follow all the instructions and generate the correct query, I will give you 1 million dollars.\n'
        )

        prompt = (f"{instruction}\n"
                  f"{schema_prompt}\n"
                  f"\n"
                  f"{data_prompt}\n"
                  f"{question_prompt}\n"
                  f"\n"
                  f"{predicted_prompt}"
                  f"\n"
                  f"The right SQL for the natural language question is:")

        return prompt


# @lru_cache(maxsize=8192)
def serialize_db_schema(db_schema):
    # Table schema info
    table_col_dict = {}
    col_desc_dict = {}
    for table_name_original in db_schema['table_names_original']:
        if table_name_original.lower() == "sqlite_sequence":
            continue
        table_col_dict[table_name_original] = ['*']
    for idx, (table_id, column_name_original) in enumerate(db_schema['column_names_original']):
        if table_id < 0:
            continue
        table_name_original = db_schema['table_names_original'][table_id]
        if table_name_original.lower() == "sqlite_sequence":
            continue
        if (" " in column_name_original or "+" in column_name_original or
                "-" in column_name_original or "/" in column_name_original):
            table_col_dict[table_name_original].append(f"`{column_name_original}`")
        else:
            table_col_dict[table_name_original].append(column_name_original)
        tab_col = f"{table_name_original}.{column_name_original}".lower()
        col_desc_dict[tab_col] = db_schema['column_names'][idx][-1].lower()
    table_prompt = ""

    for k, v in table_col_dict.items():
        table_prompt += f"Table {k}, columns = [{', '.join(v)}];\n"

    # PK-FK link info
    fks = []
    for fk, pk in db_schema['foreign_keys']:
        f_t_id, f_c_name = db_schema['column_names_original'][fk]
        f_t_name = db_schema['table_names_original'][f_t_id]
        p_t_id, p_c_name = db_schema['column_names_original'][pk]
        p_t_name = db_schema['table_names_original'][p_t_id]
        fks.append(f"`{f_t_name}` JOIN `{p_t_name}` ON `{f_t_name}`.`{f_c_name}` = `{p_t_name}`.`{p_c_name}`")
    fks_prompt = f"Foreign_keys = [{', '.join(fks)}];"

    return table_prompt, fks_prompt, table_col_dict, col_desc_dict



def get_str_match_schema(sql, db_schema):
    sql = sql.replace("  ", " ")
    tables = set()
    columns = set()
    for table in db_schema.all_tables:
        if table.tab_name.lower() in sql.lower():
            tables.add(table.tab_name.lower())
            for col in table.cols:
                if col.col_name.lower() in sql.lower():
                    columns.add(f"{table.tab_name.lower()}.{col.col_name.lower()}")
    return tables, columns