# -*- coding: utf-8 -*-
# @Time    : 2023/4/26 14:27
# @Author  :
# @Email   :  
# @File    : config.py
# @Software: PyCharm

# Set OpenAI API Key
OPENAI_KEY = "sk-"
TRANS_OPENAI_KEY = ""
DEEPSEEK_KEY = "sk-"

LLAMA_KEY = ""

MOSS_KEY = ""
PALM_KEY = ""

if isinstance(OPENAI_KEY, list):
    OPENAI_KEY *= 9999999
