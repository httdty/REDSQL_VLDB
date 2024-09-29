# -*- coding: utf-8 -*-
# @Time    : 2024/4/5 15:29
# @Author  :
# @Email   :  
# @File    : report.py
# @Software: PyCharm
from enum import Enum, auto


class StrEnum(Enum):
    def __str__(self):
        return self.value


class BugLevel(StrEnum):
    ERROR = 'ERROR'
    WARNING = 'WARNING'
    INFO = 'INFO'


class Report:
    def __init__(self, bug_level: BugLevel, location: str, description: str):
        self.level = bug_level
        self.location = location
        self.description = description

    def show(self):
        return f"[{str(self.level)}]: \"{self.location}\", {self.description}"

    def __str__(self):
        return f"[{str(self.level)}]: \"{self.location}\", {self.description}"
