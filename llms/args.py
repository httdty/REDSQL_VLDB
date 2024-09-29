# -*- coding: utf-8 -*-
# @Time    : 2023/9/20 16:51
# @Author  :
# @Email   :  
# @File    : args.py
# @Software: PyCharm
from typing import Optional
from dataclasses import dataclass, field

SUPPORTED_MODEL = {'gpt-4', 'gpt-3.5-turbo', 'text-davinci-003', 'main/text-bison-001'}

@dataclass
class LLMArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name: str = field(
        metadata={
            "help": "Model name",
            "validator": lambda x: x in SUPPORTED_MODEL
        }
    )
    gpu: bool = field(
        default=False,
        metadata={"help": "Enable GPU"},
    )

