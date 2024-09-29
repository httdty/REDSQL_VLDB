# -*- coding: utf-8 -*-
# @Time    : 2023/4/10 09:12
# @Author  :
# @Email   :  
# @File    : few_shot.py
# @Software: PyCharm
import copy
import os
import random
from functools import lru_cache
from typing import List, Dict

import tiktoken

from schema_prune.bridge_content_encoder import get_column_picklist, is_number, is_int


class FewShotPrompter:
    def __init__(self, demonstrations: List, max_length: int = 2048, **kwargs):
        self.demonstrations = [self._demonstration_init(demo) for demo in demonstrations]
        self.max_length = max_length
        self.kwargs = kwargs
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.kwargs.get("model_name", 'gpt-3.5-turbo'))
        except KeyError:
            self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')

    @lru_cache(maxsize=256, typed=False)
    def get_token_len(self, input_str: str):
        return len(self.tokenizer.encode(input_str))

    def _demonstration_init(self, demo):
        demo['prompt'] = self._serialize(demo, is_inference=False)
        return demo

    def _serialize(self, ins: Dict, is_inference: bool = True) -> str:
        raise NotImplementedError

    def get_prompt(self, ins):
        # Init
        task_desc = "Text2SQL task: Give you database schema and NL question, " \
                    "generate an executable sqlite query for me.\n\n"
        ins_prompt = self._serialize(ins)
        icl_prompt = ""
        budget = self.max_length - self.get_token_len(ins_prompt) - self.get_token_len(task_desc)

        # Ranking demonstrations here
        choices = self._ranking(ins)

        # Budget limit
        patience = 5
        while patience > 0:
            idx = choices.pop(0)
            demo_prompt = self.demonstrations[idx]['prompt'] + "\n\n"
            demo_len = self.get_token_len(demo_prompt)
            if demo_len > budget:
                patience -= 1
                continue
            else:
                budget -= demo_len
                # icl_prompt += demo_prompt
                icl_prompt = demo_prompt + icl_prompt  # reverse ranking
        return task_desc + icl_prompt + ins_prompt

    def _ranking(self, ins) -> List[int]:
        raise NotImplementedError


class RandomFewShotPrompter(FewShotPrompter):
    def __init__(self, demonstrations: List, **kwargs):
        self.db_dir = kwargs['db_dir']
        super().__init__(demonstrations, **kwargs)

    def _serialize(self, ins: Dict, is_inference: bool = True):
        db_path = os.path.join(self.db_dir, ins['db_id'], f"{ins['db_id']}.sqlite")

        # DB info
        table_lines = []
        for table in ins['db_schema']:
            table_line = f"Here are some typical values for each column in table {table['table_name_original']}:\n" \
                         f"Table: {table['table_name_original']}\n"
            for c_idx, column in enumerate(table['column_names_original']):
                db_contents = table['db_contents'][c_idx]
                if len(db_contents) < 3:
                    vals = get_column_picklist(
                        table_name=table['table_name_original'], column_name=column, db_path=db_path
                    )
                    db_contents += vals
                table_line += f"{column}: {' , '.join([str(v) for v in db_contents[:3]])}\n"
            table_lines.append(table_line)

        if len(ins['fk']) > 0:
            fk_line = "The foreign keys:\n"
            for fk in ins['fk']:
                line = f"{fk['source_table_name_original']}.{fk['source_column_name_original']} = " \
                       f"{fk['target_table_name_original']}.{fk['target_column_name_original']}\n"
                fk_line += line
            # table_lines.append(fk_line)

        db_info = '\n'.join(table_lines)
        db_info = f"\n'''{db_info}'''"
        nl_info = f"The question is '{ins['question']}';"
        if is_inference:
            answer_info = "The SQL query is: "
        else:
            answer_info = f"The SQL query is: {ins['sql']};"
        prompt = f"{db_info}\n{nl_info}\n{answer_info}"
        return prompt

    def _ranking(self, ins) -> List[int]:
        idx_list = list(range(len(self.demonstrations)))
        random.shuffle(idx_list)
        return idx_list


def serialize_raw_schema(schema: Dict):
    for c_i, (t_i, col) in enumerate(schema['column_names_original']):
        if t_i < 0:
            continue
        tab = schema['table_names_original'][t_i]


class OrangeFewShotPrompter(FewShotPrompter):
    def __init__(self, demonstrations: List, **kwargs):
        self.db_dir = kwargs['db_dir']
        self.explain_db = kwargs['explain_db']
        self.explain_nl = kwargs['explain_nl']
        self.explain_sql = kwargs['explain_sql']
        self.db_question = {}
        self._preprocess_demo(demonstrations)
        super().__init__(demonstrations, **kwargs)

    def _preprocess_demo(self, demonstrations):
        for demo in demonstrations:
            self.db_question.setdefault(demo['db_id'], [])
            self.db_question[demo['db_id']].append(demo)

    def enhance_by_task(self, dev):
        for d in dev:
            self.db_question.setdefault(d['db_id'], [])
            self.db_question[d['db_id']].append(d)

    def _serialize_query(self, ins, is_inference):

        nl_info = f"The question is '{ins['question']}';"
        if is_inference:
            answer_info = "The SQL query is: "
        else:
            answer_info = f"The SQL query is: {ins['sql']};"  # Format the SQL
        return nl_info + "\n" + answer_info

    def _serialize(self, ins: Dict, is_inference: bool = True):
        db_path = os.path.join(self.db_dir, ins['db_id'], f"{ins['db_id']}.sqlite")

        # DB info
        table_lines = []
        tables = copy.deepcopy(ins['db_schema'])
        # random.shuffle(tables)

        # for table in tables:
        #     cols = ", ".join(["*"] + table['column_names_original'])
        #     table_line = f"Table: {table['table_name_original']} = [{cols}]\n"
        #     head_line = "|"
        #     data_num = 3 if is_inference else 2
        #     data_lines = ["|" for _ in range(data_num)]
        #
        #     for c_idx, column in enumerate(table['column_names_original']):
        #         head_line += column + "|"
        #         db_contents = table['db_contents'][c_idx]
        #         vals = get_column_picklist(
        #             table_name=table['table_name_original'], column_name=column, db_path=db_path
        #         ) + ["NULL"] * 3
        #         db_contents += vals
        #         for line_num in range(data_num):
        #             data_lines[line_num] += str(db_contents[line_num]) + "|"
        #     data_line = "\n".join(data_lines)
        #     table_line += f"{head_line}\n{data_line}\n"
        #     table_lines.append(table_line)

        for table in tables:
            # table_line = f"Here are some typical values for each column in table '{table['table_name_original']}':\n" \
            # f"Table: {table['table_name_original']}\n"
            cols = ", ".join(["*"] + table['column_names_original'])
            table_line = f"Table: {table['table_name_original']} = [{cols}]\n"
            for c_idx, column in enumerate(table['column_names_original']):
                db_contents = table['db_contents'][c_idx]
                vals = get_column_picklist(
                    table_name=table['table_name_original'], column_name=column, db_path=db_path
                )
                db_contents += vals
                if is_inference:
                    table_line += f"{column}: {' , '.join([str(v) for v in db_contents[:3]])}\n"
                else:
                    table_line += f"{column}: {' , '.join([str(v) for v in db_contents[:2]])}\n"
            table_lines.append(table_line)
        if len(ins['fk']) > 0:
            fk_line = "The foreign keys:\n"
            for fk in ins['fk']:
                line = f"{fk['source_table_name_original']}.{fk['source_column_name_original']} = " \
                       f"{fk['target_table_name_original']}.{fk['target_column_name_original']}\n"
                fk_line += line
            table_lines.append(fk_line)
        db_info = '\n'.join(table_lines)
        db_info = f"'''\n{db_info}'''"

        question_context = ""
        if ins['db_id'] in self.db_question and not is_inference:
            question_context = "Related questions are:\n"
            workload = [
                {
                    'question': q['question'],
                    'db_id': q['db_id'],
                    'sql': q['sql'],
                    'explain_nl': q['explain_nl'],
                    'explain_sql': q['explain_sql'],
                    'explain_db': q['explain_db'],
                } for q in self.db_question[ins['db_id']][:10]
            ]
            random.shuffle(workload)
            for demo in workload:
                if demo['question'] in ins['question']:
                    continue
                question_context += "\n" + self._serialize_query(demo, is_inference) + "\n"

        # nl_info = f"The question is '{ins['question']}';"
        # if is_inference:
        #     answer_info = "The SQL query is: "
        # else:
        #     answer_info = f"The SQL query is: {ins['sql']};"  # Format the SQL

        query = self._serialize_query(ins, is_inference)

        prompt = f"{db_info}\n{query}{question_context}"
        return prompt

        # table_lines = []
        # for table in ins['db_schema']:
        #     # table_line = f"Here are some typical values for each column in table {table['table_name_original']}:\n"
        #     table_line = ""
        #
        #     table_line += f"Table {table['table_name_original']}"
        #     table_line += (f", columns = "
        #                    f"[*, {', '.join([c for _, c in enumerate(table['column_names_original'])])}]")
        #
        #     table_lines.append(table_line)
        #
        # if len(ins['fk']) > 0:
        #     fk_line = "foreign_keys = ["
        #     fk_lines = []
        #     for fk in ins['fk']:
        #         line = f"{fk['source_table_name_original']}.{fk['source_column_name_original']} = " \
        #                f"{fk['target_table_name_original']}.{fk['target_column_name_original']}"
        #         fk_lines.append(line)
        #     fk_line += ", ".join(fk_lines) + "]"
        #     table_lines.append(fk_line)
        # if len(ins['pk']) > 0:
        #     pk_line = "primary_keys = ["
        #     pk_lines = []
        #     for pk in ins['pk']:
        #         pk_lines.append(f"{pk['table_name_original']}.{pk['column_name_original']}")
        #     pk_line += ", ".join(pk_lines) + "]"
        #     table_lines.append(pk_line)
        #
        # # Explain DB
        # if self.explain_db and is_inference:
        #     # explain_db_lines = []
        #     for table in ins['db_schema']:
        #         table_explain = ""
        #         if self.explain_db and is_inference:
        #             table_explain = ins['explain_db'].get(table['table_name_original'].replace(" ", "_"), "")
        #             if "includ" in table_explain:
        #                 table_explain = table_explain[:table_explain.index("includ") - 1]
        #                 table_explain = table_explain.strip(",")
        #         if table_explain:
        #             table_explain = f" ({table_explain})"
        #         table_line = (f"Table {table['table_name_original'].replace(' ', '_')}{table_explain.lower()}, "
        #                       f"typical values for each columns:")
        #         for c_idx, column in enumerate(table['column_names_original']):
        #             db_contents = table['db_contents'][c_idx]
        #             if len(db_contents) < 3:
        #                 vals = get_column_picklist(
        #                     table_name=table['table_name_original'], column_name=column, db_path=db_path
        #                 )
        #                 db_contents += vals
        #             vals = db_contents[:3]
        #             for idx, v in enumerate(vals):
        #                 if is_number(v):
        #                     if is_int(v):
        #                         vals[idx] = int(v.replace(",", ""))
        #                     else:
        #                         vals[idx] = float(v.replace(",", ""))
        #             # table_line += f"\n{column.lower()}: {' , '.join([str(v) for v in db_contents[:3]])}"
        #             table_line += f"\n{column.lower()}: {vals}"
        #         table_lines.append(table_line)
        #
        # db_info = "'''\nDatabase:\n" + '\n'.join(table_lines) + "'''"
        # db_info = f"\n{db_info}"
        #
        # # Explain NL
        # nl_info = f"### Question: {ins['question']}"
        # if is_inference:
        #     if self.explain_nl:
        #         answer_info = "### SQL query:"
        #
        #         # answer_info = "### Question rewrite: "
        #     else:
        #         answer_info = "### SQL query:"
        #
        # else:
        #     # if self.explain_nl:
        #     #     nl_info += f"\n### Question rewrite: {ins['explain_nl']}"
        #     answer_info = ""
        #     # Explain SQL
        #     if self.explain_sql and self.explain_nl:
        #         answer_info += (f"Please think step by step. To solve the problem, we should: \n"
        #                         f"Question understanding based on Schema: {ins['explain_nl']}\n"
        #                         f"{ins['explain_sql']}\n")
        #     elif self.explain_sql:
        #         answer_info += (f"### Please think step by step. To solve the problems, we should: "
        #                         f"{ins['explain_sql']}\n")
        #     answer_info += f"### SQL query: {ins['sql']}"
        #
        # prompt = f"{db_info}\n{nl_info}\n{answer_info}"
        # return prompt

    def _ranking(self, ins) -> List[int]:
        # idx_list = list(range(len(self.demonstrations)))
        idx_list = list()
        added_db_id = set()
        explain_sql = any(['explain_sql' in demo for demo in self.demonstrations])
        for i, demo in enumerate(self.demonstrations):
            if demo['db_id'] in added_db_id or len(demo['prompt']) > 6800:
                continue
            if explain_sql:
                if len(demo['sql_skeleton']) > 60 and len(demo['explain_sql']) > 10:
                    idx_list.append(i)
                    added_db_id.add(demo['db_id'])
            elif len(demo['sql_skeleton']) > 60:
                idx_list.append(i)
                added_db_id.add(demo['db_id'])

        return idx_list
