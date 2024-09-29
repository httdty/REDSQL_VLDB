# -*- coding: utf-8 -*-
# @Time    : 2024/4/2 19:22
# @Author  :
# @Email   :  
# @File    : schema_item.py
# @Software: PyCharm
from typing import List


class Table:
    def __init__(self, name: str, col_names: List, pks: List, col_types: List = None):
        self.tab_name = name
        self.cols: List[Column] = []
        self.alias = set()
        for col in col_names:
            self.cols.append(Column(col, self))
        self.pks = set([pk.lower() for pk in pks])
        if col_types and len(col_types) == len(self.cols):
            for i, col_type in enumerate(col_types):
                self.cols[i].set_type(col_type)

    def get_column(self, col_name: str):
        if "." in col_name:
            tab_name, col_name = col_name.split(".", maxsplit=1)
            tab_name = tab_name.strip('`').strip('"')
            if tab_name in self.alias:
                tab_name = self.tab_name
            if tab_name.lower() != self.tab_name.lower():
                return None
        col_name = col_name.strip('`').strip('"')
        for col in self.cols:
            if col.col_name.lower() == col_name.lower() or col_name in col.alias:
                return col
        return None

    def neighbors(self):
        res = set()
        for col in self.cols:
            for fk in col.fks:
                res.add(fk.tab)
        return res

    def set_alias(self, alias):
        self.alias.add(alias)

    def is_single_column(self):
        return len(self.cols) == 1

    def is_single_value(self):
        return len(self.cols) == 1 and self.cols[0].is_list()

    def is_null(self):
        return len(self.cols) == 1 and self.cols[0].is_null()

    def __str__(self):
        return self.tab_name + ": " + ", ".join([str(col) for col in self.cols])

    def __eq__(self, other):
        if not isinstance(other, Table):
            return False
        return self.tab_name.lower() == other.tab_name.lower()

    def __hash__(self):
        return hash(str(self))

    def __len__(self):
        return len(self.cols)


class Column:
    def __init__(self, col_name: str, tab: Table):
        self.tab = tab
        # if tab_name:
        #     self.table = Table(tab_name)
        if "." in col_name:
            tab_name, col_name = col_name.split(".", maxsplit=1)
        self.col_name = col_name
        self.col_type = ""
        self.fks: List[Column] = []
        self.extended_fks: List[Column] = []
        self.alias = set()
        self.values = None
        self.has_null = None

    def __eq__(self, other):
        if not isinstance(other, Column):
            return False
        return self.col_name.lower() == other.col_name.lower() and self.tab == other.tab

    def set_alias(self, alias):
        self.alias.add(alias)

    def __str__(self):
        if self.tab:
            return self.tab.tab_name + "." + self.col_name
        else:
            return self.col_name

    def is_pk(self):
        return self.col_name.lower() in self.tab.pks

    def is_fk_with(self, col: 'Column'):
        return col in self.fks or self in col.fks or col in self.extended_fks or self in col.extended_fks

    def set_fk(self, col: 'Column'):
        if isinstance(col, Column) and col not in self.fks:
            self.fks.append(col)
            col.fks.append(self)
            self.set_extended_fks(col)

    def set_extended_fks(self, col: 'Column'):
        if col not in self.extended_fks and col != self:
            self.extended_fks.append(col)
            col.extended_fks.append(self)
        for fk in col.fks:
            if fk not in self.extended_fks and col != self:
                self.set_extended_fks(fk)

    def set_type(self, col_type: str):
        self.col_type = col_type.upper()

    def add_type(self, col_type: str):
        col_type = col_type.upper()
        if col_type and col_type in self.col_type:
            return
        if not self.col_type:
            self.col_type += f"{col_type}".upper()
        else:
            self.col_type += f"&{col_type}".upper()

    def set_values(self, values):
        self.values = values

    def is_single_value(self):
        if not isinstance(self.values, list):
            return False
        else:
            return len(self.values) == 1

    def is_list(self):
        if not isinstance(self.values, list):
            return False
        else:
            return len(self.values) == 1

    def is_null(self):
        if not isinstance(self.values, list):
            return False
        else:
            return len(self.values) == 0


    def __hash__(self):
        return hash(str(self))
