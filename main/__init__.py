# -*- coding: utf-8 -*-
# @Time    : 2023/4/10 09:11
# @Author  :
# @Email   :  
# @File    : __init__.py.py
# @Software: PyCharm
from loguru import logger
import datetime
import sys


log_file_name = "_".join(sys.argv).replace("/", "_")
logger.add(
    f"./logs/{str(datetime.datetime.now()).replace(' ', '_').split('.')[0]}_{log_file_name}"[:68] + ".txt",
    format="{time} | {level:7} | {message}"
)
