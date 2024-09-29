# -*- coding: utf-8 -*-
# @Time    : 2024/3/22 14:46
# @Author  :
# @Email   :  
# @File    : schema.py
# @Software: PyCharm
from copy import deepcopy
from typing import Dict, List
import networkx as nx

from red.parser.schema_item import Table, Column
from red.parser.utils import all_is_int, all_is_number, execute_sql, type_of_query


class Schema:
    _META_LIMIT = "LIMIT 1000"

    def __init__(self, raw_schema: Dict, db_path: str):
        self.db_path = db_path
        self.raw_schema = raw_schema
        self.db_id = raw_schema['db_id']
        self._table_names_original = raw_schema['table_names_original']
        self._column_names_original = raw_schema['column_names_original']
        self._primary_keys = raw_schema['primary_keys']
        self._foreign_keys = raw_schema['foreign_keys']
        self._column_types = raw_schema['column_types']
        self.all_tables: List[Table] = []
        self.sub_queries = {}
        self.expressions = {}
        self.values = {}
        self.query_tabs = []
        self.graph = SchemaGraph(self)
        # self.query_cols = []

        self.parse()
        self.read_db_metadata()

    def parse(self):
        pk_idx_list = []
        for pk in self._primary_keys:
            if isinstance(pk, list):
                pk_idx_list += pk
            else:
                pk_idx_list.append(pk)
        table_idx = 0
        cols = []
        col_types = []
        pks = []
        for i, (idx, col) in enumerate(self._column_names_original):
            if idx < 0:
                continue
            if idx != table_idx:
                self.all_tables.append(Table(self._table_names_original[table_idx], cols, pks, col_types))
                table_idx = idx
                cols = []
                col_types = []
                pks = []
            cols.append(col)
            col_types.append(self._column_types[i])
            if i in pk_idx_list:
                pks.append(col)
        if cols:
            self.all_tables.append(Table(self._table_names_original[table_idx], cols, pks, col_types))

        for fks in self._foreign_keys:
            col_1, col_2 = self._column_names_original[fks[0]], self._column_names_original[fks[1]]
            tab_1, tab_2 = self._table_names_original[col_1[0]], self._table_names_original[col_2[0]]
            col_1, col_2 = col_1[1], col_2[1]
            tab_1, tab_2 = self.get_table(tab_1), self.get_table(tab_2)
            assert isinstance(tab_1, Table)
            assert isinstance(tab_2, Table)
            col_1, col_2 = tab_1.get_column(col_1), tab_2.get_column(col_2)
            col_1.set_fk(col_2)
            col_2.set_fk(col_1)

    def read_db_metadata(self):
        for tab_name in self._table_names_original:
            # Parse column type
            sql = f"PRAGMA table_info(`{tab_name}`)"
            _, columns = execute_sql(self.db_path, sql)
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                column = self.get_all_column(f"`{tab_name}`.`{col_name}`")
                if col_type == "TEXT":
                    col_type = self._is_number_text(column)
                column.add_type(col_type)
                sql = f"SELECT typeof(`{col_name}`) FROM `{tab_name}` WHERE `{col_name}` IS NOT NULL {self._META_LIMIT};"
                types = type_of_query(self.db_path, sql)
                if len(types) > 1:
                    print(types)
                    print("WARNING: more than one data type in column")
                if types:
                    column.add_type(types.pop())
                sql = f"SELECT COUNT(*) FROM `{tab_name}` WHERE `{col_name}` IS NULL;"
                _, res = execute_sql(self.db_path, sql)
                column.has_null = res[0][0] > 0
                if "INTEGER" in column.col_type or "REAL" in column.col_type:
                    sql = f"SELECT DISTINCT `{col_name}` FROM `{tab_name}` WHERE `{col_name}` IS NOT NULL {self._META_LIMIT};"
                    _, values = execute_sql(self.db_path, sql)
                    values = set([v[0] for v in values])
                    column.values = values
                else:
                    sql = f"SELECT DISTINCT `{col_name}` FROM `{tab_name}` WHERE `{col_name}` IS NOT NULL AND LENGTH(`{col_name}`) < 50;"
                    _, values = execute_sql(self.db_path, sql)
                    values = set([v[0] for v in values])
                    column.values = values

            # Parse FKs
            sql = f"PRAGMA foreign_key_list(`{tab_name}`)"
            _, fks = execute_sql(self.db_path, sql)
            for fk in fks:
                col_name = fk[3]
                f_tab = fk[2]
                f_col = fk[4]
                column = self.get_all_column(f"`{tab_name}`.`{col_name}`")
                f_column = self.get_all_column(f"`{f_tab}`.`{f_col}`")
                column.set_fk(f_column)

    def _is_number_text(self, column: Column):
        if column.col_type != 'TEXT':
            return column.col_type
        table = column.tab
        sql = f"SELECT `{column.col_name}` FROM `{table.tab_name}` WHERE `{column.col_name}` IS NOT NULL {self._META_LIMIT};"
        _, values = execute_sql(self.db_path, sql)
        values = [value[0] for value in values]
        if all_is_int(values):
            return "INTEGER"
        elif all_is_number(values):
            return "REAL"
        return column.col_type

    def add_query_table(self, table):
        from red.parser.red_parser import Query

        if isinstance(table, Table):
            self.query_tabs.append(deepcopy(table))
        elif isinstance(table, Query):
            self.query_tabs.append(deepcopy(table.exec_table))
        else:
            raise ValueError("Do not support such table")

    def get_table(self, table_name: str) -> Table:
        table_name = table_name.strip("`").strip('"')
        if table_name in self.sub_queries:
            return self.sub_queries[table_name].exec_table
        for table in self.query_tabs:
            if table_name.upper() in table.alias or table_name.lower() == table.tab_name.lower():
                return table
        for table in self.all_tables:
            if table_name.upper() in table.alias or table_name.lower() == table.tab_name.lower():
                return table

    def get_all_table(self, table_name: str) -> Table:
        table_name = table_name.strip("`").strip('"')
        if table_name in self.sub_queries:
            return self.sub_queries[table_name].exec_table
        for table in self.all_tables:
            if table_name.upper() in table.alias or table_name.lower() == table.tab_name.lower():
                return table

    def get_column(self, col_name: str):
        if "." in col_name:
            tab_name, col_name = col_name.split(".", maxsplit=1)
            col_name = col_name.strip("'")
            table = self.get_table(tab_name)
            if table is None:
                return None
            return table.get_column(col_name)

        # col_name.strip("`")
        for table in self.query_tabs:
            col = table.get_column(col_name)
            if col:
                return col
        if col_name.upper() in self.expressions:
            return self.expressions[col_name.upper()]
        elif col_name.upper() in self.sub_queries:
            return self.sub_queries[col_name.upper()]
        return None

    def get_all_column(self, col_name: str) -> Column:
        if "." in col_name:
            tab_name, col_name = col_name.split(".", maxsplit=1)
            table = self.get_table(tab_name)
            return table.get_column(col_name)
        for table in self.all_tables:
            col = table.get_column(col_name)
            if col:
                return col

    def add_alias(self, alias: str, target):
        from red.parser.red_parser import Query
        from red.parser.red_parser import Expression

        alias = alias.upper()
        if isinstance(target, Table) or isinstance(target, Column):
            target.set_alias(alias)
        elif isinstance(target, Query):
            self.sub_queries[alias] = target
        elif isinstance(target, Expression):
            self.expressions[alias] = target
        else:
            self.values[alias] = target

    def choose_table(self, tab: Table):
        self.query_tabs.append(tab)

    def is_schema(self, name):
        if self.get_table(name):  # Do not support Subquery and Expression
            return True
        if self.get_column(name):
            return True
        if name in self.expressions:
            return True
        elif name in self.sub_queries:
            return True
        return False

class SchemaGraph:
    def __init__(self, schema: Schema):
        self.schema = schema
        self.graph = nx.Graph()
        self.build_graph()

    def build_graph(self):
        for table in self.schema.all_tables:
            self.graph.add_node(table.tab_name)
        schema = self.schema.raw_schema

        for fks in schema['foreign_keys']:
            col_1_idx, col_2_idx = fks
            col_1 = schema['column_names_original'][col_1_idx]
            col_2 = schema['column_names_original'][col_2_idx]
            tab_1 = schema['table_names_original'][col_1[0]]
            tab_2 = schema['table_names_original'][col_2[0]]
            self.graph.add_edge(tab_1, tab_2, relationship=f"{tab_1}.{col_1[1]} = {tab_2}.{col_2[1]}")

    def find_minimum_spanning_tree(self, tables):
        subgraph = self.graph.subgraph(tables)
        mst = nx.minimum_spanning_tree(subgraph)
        if mst:
            start_node = list(mst.nodes)[0]
        else:
            return ""
        path = []
        visited = set()

        def dfs(node):
            visited.add(node)
            neighbors = list(mst[node])
            for neighbor in neighbors:
                if neighbor not in visited:
                    edge_data_ = mst[node][neighbor]
                    path.append((node, neighbor, edge_data_))
                    dfs(neighbor)

        dfs(start_node)

        path_str = ""
        for i, (node1, node2, edge_data) in enumerate(path):
            if i == 0:
                path_str += f"{node1}"
            path_str += f" JOIN {node2} ON {edge_data['relationship']}"

        return path_str
