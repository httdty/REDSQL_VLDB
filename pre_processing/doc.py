# -*- coding: utf-8 -*-
# @Time    : 2024/1/14 19:55
# @Author  :
# @Email   :  
# @File    : doc.py
# @Software: PyCharm
import argparse
import json
import sqlite3
from typing import List, Dict

import pandas as pd
from tqdm import tqdm
from ydata_profiling import ProfileReport

from bug_fix.consistency import get_cursor_from_path
from llms import model_init
from pre_processing.db_utils import is_number
from red.red import serialize_db_schema


class Documentation:
    def __init__(self, schema_file: str, db_dir: str, model_name: str = 'gpt-4-32k', **kwargs):
        self.schemas = {}
        self.db_dir = db_dir
        self.kwargs = kwargs
        self._init_schema(schema_file)
        self.llm = model_init(model_name)

    def _init_schema(self, schema_file):
        with open(schema_file, 'r') as f:
            schema_list = json.load(f)
        for schema in tqdm(schema_list):
            table_prompt, fks_prompt, table_dict, col_desc_dict = serialize_db_schema(schema)
            schema_prompt = table_prompt + fks_prompt
            schema['schema_prompt'] = schema_prompt
            schema['table_dict'] = table_dict
            schema['col_desc_dict'] = col_desc_dict
            schema['data'] = {}
            schema['profile'] = {}
            for tab in schema['table_dict']:
                print()
                sqlite_path = f"{self.db_dir}/{schema['db_id']}/{schema['db_id']}.sqlite"
                connection = sqlite3.connect(sqlite_path)
                connection.text_factory = handle_invalid_utf8
                query = f"SELECT * FROM `{tab}`"
                try:
                    df = pd.read_sql_query(query, connection)
                    df.columns = df.columns.str.lower()
                    profile = ProfileReport(df, title="Profiling Report", minimal=True,
                                            correlations={"auto": {"calculate": False}})
                    json_data = profile.to_json()
                    profile = json.loads(json_data)
                except Exception as e:
                    print(e)
                    profile = {}
                for col in schema['table_dict'][tab]:
                    col = col.strip("`")
                    if col == '*':
                        continue
                    values = []
                    cursor = get_cursor_from_path(sqlite_path)
                    cursor.execute(f"SELECT `{col}` FROM `{tab}` WHERE `{col}` IS NOT NULL ORDER BY `{col}` LIMIT 2;")
                    vals = cursor.fetchall()
                    values += [val[0] if is_number(val[0]) else str(val[0])[:20] for val in vals]

                    cursor.execute(f"SELECT COUNT(*) FROM `{tab}` WHERE `{col}` IS NULL;")
                    vals = cursor.fetchall()
                    if vals[0][0] > 0:
                        values.append(None)

                    cursor.execute(
                        f"SELECT `{col}` FROM `{tab}` WHERE `{col}` IS NOT NULL ORDER BY `{col}` DESC LIMIT 2;")
                    vals = cursor.fetchall()
                    values += [val[0] if is_number(val[0]) else str(val[0])[:20] for val in vals]

                    schema['data'][f'{tab}.{col}'.lower()] = values

                    c_p = profile.get("variables", {}).get(col.lower(), {})
                    p = ""
                    if c_p and c_p['type'] != 'Unsupported':
                        if c_p['is_unique']:
                            p += f"unique column with {c_p['n_distinct']} rows, "
                        else:
                            most_common = max(c_p['value_counts_index_sorted'],
                                              key=c_p['value_counts_index_sorted'].get)
                            if c_p['type'] == 'Numeric':
                                p += f"most common value: {most_common}, "
                            else:
                                p += f"most common value: '{most_common}', "
                        if c_p['type'] == 'Numeric':
                            p += f"Numeric column, min value: {c_p['min']}, max value: {c_p['max']}. "
                        elif c_p['type'] == 'Categorical':
                            if c_p['n_distinct'] < 10:
                                all_possible_value = list(c_p['value_counts_index_sorted'].keys())
                                p += f"Categorical column, all possible values: {all_possible_value}. "
                        elif c_p['type'] == 'Text':
                            p += f"Text column. "
                        elif c_p['type'] == 'DateTime':
                            p += f"DateTime column, min value: {c_p['min']}, max value: {c_p['max']}. "
                    schema['profile'][f'{tab}.{col}'.lower()] = p

            pks = []
            fks = {}
            if schema:
                for pk in schema['primary_keys']:
                    if isinstance(pk, list):
                        for pk_i in pk:
                            tab_id, col = schema['column_names_original'][pk_i]
                            tab = schema['table_names_original'][tab_id]
                            pks.append(f"{tab}.{col}".lower())
                    else:
                        tab_id, col = schema['column_names_original'][pk]
                        tab = schema['table_names_original'][tab_id]
                        pks.append(f"{tab}.{col}".lower())
                for fk, pk in schema['foreign_keys']:
                    f_t_id, f_c_name = schema['column_names_original'][fk]
                    f_t_name = schema['table_names_original'][f_t_id]
                    p_t_id, p_c_name = schema['column_names_original'][pk]
                    p_t_name = schema['table_names_original'][p_t_id]
                    fks[f"{f_t_name}.{f_c_name}".lower()] = f"`{p_t_name}`.`{p_c_name}`"
            schema['pks'] = pks
            schema['fks'] = fks
            schema['annotation'] = {}
            self.schemas[schema['db_id']] = schema

    def documentation(self, db_id):
        schema = self.schemas[db_id]
        tab_cols = []
        table_data_prompt = []
        tables = list(schema['table_dict'].keys())

        for idx, tab in enumerate(tables):
            instruction = (f"The following information includes the schema of Database `{db_id}` "
                           f"and a sample display of data from Table `{tab}`. "
                           f"Please provide meaning for each column in `{tab}`.\n"
                           f"Notice that we want to include the business meaning of the KEYs(Primary Key or Foreign Key)."
                           # f"Do not need to include the values in the annotation."
                           )
            data_prompt = f"Table `{tab}`:\n"
            cols = schema['table_dict'][tab]
            for col in cols:
                col = col.strip("`")
                if col == '*':
                    continue
                tab_col = f'{tab}.{col}'.lower()
                tab_cols.append(tab_col)

                col_modifier = ""
                if tab_col in schema['fks']:
                    col_modifier = f" (Foreign Key, link to {schema['fks'][tab_col]})"
                elif tab_col in schema['pks']:
                    col_modifier = " (Primary Key)"
                col_modifier += f" # {schema['col_desc_dict'][tab_col]}"
                col_profile = schema['profile'].get(tab_col, "")
                data_prompt += (f"`{col}`{col_modifier}, {col_profile}typical values: "
                                f"{schema['data'][tab_col]};\n")
            table_data_prompt.append(data_prompt)
            if (len(schema['schema_prompt'] + data_prompt) > 20000 or tab == tables[-1] or
                    len(schema['table_dict'][tables[idx + 1]]) > 60):
                leading_prompt = (
                    f"The format is `table_name`.`column_name`: `meaning`;\n"
                    f"For example, `stu`.`stu_age`: The age of the student;\n"
                    f"Each column annotation should be displayed in a separate line.\n"
                    f"The meaning of each column of each table mentioned above are:\n")
                table_data_prompt.sort(key=lambda x: len(x))
                data_prompt = ("The semantic of the following columns of each table should be annotated:\n"
                               + '\n'.join(table_data_prompt))
                prompt = (f"{instruction}\n"
                          f"\n"
                          f"{schema['schema_prompt']}\n"
                          f"\n"
                          f"{data_prompt}\n"
                          f"{leading_prompt}")
                print(prompt)
                response = self.llm.infer(prompt_list=[prompt], n=10)[0]
                annotation_dict = self.parse_res(tab_cols, response)
                self.schemas[db_id]['annotation'].update(annotation_dict)
                while len(annotation_dict) != len(tab_cols):
                    # print("[ERROR] Column annotation lost")
                    annotated = ""
                    for k, v in annotation_dict.items():
                        tab, col = k.split(".", maxsplit=1)
                        annotated += f"`{tab}`.`{col}`: {v}\n"
                    rest_cols = []
                    for col in tab_cols:
                        if col not in annotation_dict:
                            tab, col = col.split(".", maxsplit=1)
                            rest_cols.append(f"`{tab}`.`{col}`")
                    rest_prompt = f"Please keep annotating the cols: {rest_cols}\n"
                    rest_prompt = (f"{prompt}\n"
                                   f"{annotated}\n"
                                   f"{rest_prompt}")
                    print(rest_prompt)
                    rest_annotation_dict = self.parse_res(tab_cols, self.llm.infer(prompt_list=[rest_prompt], n=8)[0])
                    annotation_dict.update(rest_annotation_dict)
                # if len(annotation_dict) != len(tab_cols):
                #     print("[ERROR] Column annotation lost")
                self.schemas[db_id]['annotation'].update(annotation_dict)
                tab_cols = []
                table_data_prompt = []
        print("prompt_length:", self.llm.prompt_length)
        print("completion_length:", self.llm.completion_length)
        # print(self.schemas[db_id]['annotation'])

    @property
    def annotation(self):
        res = {}
        # for db_id in ['concert_singer',
        #               'pets_1',
        #               'car_1',
        #               'flight_2',
        #               'employee_hire_evaluation',
        #               'cre_Doc_Template_Mgt',
        #               'course_teach',
        #               'museum_visit',
        #               'wta_1',
        #               'battle_death',
        #               'student_transcripts_tracking',
        #               'tvshow',
        #               'poker_player',
        #               'voter_1',
        #               'world_1',
        #               'orchestra',
        #               'network_1',
        #               'dog_kennels',
        #               'singer',
        #               'real_estate_properties']:
        for db_id in self.schemas:
            if not self.schemas[db_id]['annotation']:
                self.documentation(db_id)
            res[db_id] = self.schemas[db_id]['annotation']
            print(json.dumps(res[db_id], indent=2))
            print()
        return res

    @staticmethod
    def parse_res(tab_cols: List[str], llm_res_list: List[str]) -> Dict:
        full_dict = {}
        for res in llm_res_list:
            lines = res.split("\n")
            for line in lines:
                if not line.strip() or ":" not in line:
                    continue
                col, annotation = line.split(":", maxsplit=1)
                if len(annotation.strip()) > 200:
                    continue
                if col.strip()[0] in {"-", "*"}:
                    col = col.strip()[1:].strip()
                if "." not in col:
                    continue
                tab, col = col.split(".", maxsplit=1)
                if "`" in col:
                    col = col.replace("`", "")
                if "`" in tab:
                    tab = tab.replace("`", "")
                tab_col = f"{tab}.{col}".lower()
                if tab_col in tab_cols:
                    if (tab_col not in full_dict or
                            (tab_col in full_dict and 60 > len(annotation.strip()) > len(full_dict[tab_col]))):
                        full_dict[tab_col] = annotation.strip()
            # if len(annotation_dict) == len(tab_cols):
            #     return annotation_dict
        return full_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # LLMs config
    parser.add_argument("--model_name",
                        type=str,
                        required=True,
                        help="Model name")
    # EXP file path
    parser.add_argument("--output_file",
                        type=str,
                        default="./annotation.json",
                        help="Output annotation")
    parser.add_argument("--table_file",
                        type=str,
                        default="./datasets/bird/dev_tables.json",
                        help="tables file")
    parser.add_argument("--db_dir",
                        type=str,
                        default="./datasets/bird/dev_database",
                        help="db_dir")

    args_ = parser.parse_args()
    return args_


def handle_invalid_utf8(text):
    try:
        return text.decode('utf-8')
    except UnicodeDecodeError:
        return text.decode('utf-8', 'ignore')


def main():
    args = parse_args()
    doc = Documentation(args.table_file, args.db_dir, model_name=args.model_name)
    annotation = doc.annotation
    with open(args.output_file, 'w') as f:
        json.dump(annotation, f, indent=2)


if __name__ == '__main__':
    main()
