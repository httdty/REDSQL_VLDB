# -*- coding: utf-8 -*-
# @Time    : 2023/4/10 09:13
# @Author  :
# @Email   :  
# @File    : __init__.py.py
# @Software: PyCharm
from llms.deepseek import DeepSeek
from llms.llama import LLAMA
from llms.openai import Openai
from llms.openai_trans import OpenaiTrans
# from llms.palm import PaLM
from llms.moss import Moss


def model_init(model_name: str, **kwargs):
    if model_name in OpenaiTrans.models:
        return OpenaiTrans(model_name, **kwargs)
    elif model_name.startswith("gpt"):
        return Openai(model_name, **kwargs)
    elif model_name in DeepSeek.models:
        return DeepSeek(model_name, **kwargs)
    elif model_name in LLAMA.models:
        return LLAMA(model_name, **kwargs)
    # elif model_name in PaLM.main:
    #     return PaLM(model_name, **kwargs)
    elif model_name == "moss":
        return Moss(model_name, **kwargs)
