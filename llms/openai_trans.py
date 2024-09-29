# -*- coding: utf-8 -*-
# @Time    : 2024/6/15 13:10
# @Author  :
# @Email   :  
# @File    : openai_trans.py
# @Software: PyCharm
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import openai
from openai import AzureOpenAI

from llms.llm import LLM
from llms.config import TRANS_OPENAI_KEY


class OpenaiTrans(LLM):
    models = {'gpt35-turbo'}

    def __init__(self, name, interval=0, **kwargs):
        self.client = AzureOpenAI(
            api_key=TRANS_OPENAI_KEY,
            api_version="2024-02-01",
            azure_endpoint= "https://transwarp-sc.openai.azure.com/"
        )
        # openai.api_type = "azure"
        # openai.api_base = "https://transwarp-sc.openai.azure.com/"
        # openai.api_version = "2023-07-01-preview"
        # openai.api_key = TRANS_OPENAI_KEY
        super().__init__(name)
        self.api = self.client.chat.completions.create
        self.interval = interval

    def __infer_one(self, prompt, kwargs):
        response = None
        if not prompt:
            return response
        self.count += 1
        while not response:
            print("Transwarp", end="")
            try:
                if self.name in self.models:
                    response = self.api(
                        model=self.name,
                        messages=[
                            {"role": "system", "content": prompt}
                        ],
                        # temperature=0,
                        # max_tokens=256,
                        # top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        timeout=120,
                        **kwargs
                    )
            except Exception as e:
                print(e)
                print("Prompt:", prompt[:1000])
                print("Prompt Len:", len(prompt))
                time.sleep(12)
                response = None

        return response

    def infer(self, prompt_list, **kwargs):
        assert isinstance(prompt_list, list), "Please make sure the input is a list of str"
        res = []
        with ThreadPoolExecutor(max_workers=min(len(prompt_list), 32)) as pool:
            response_list = pool.map(self.__infer_one, prompt_list, repeat(kwargs))
            for response in response_list:
                if not response:
                    res.append("")
                elif self.name in self.models:
                    res.append([choice.message.content for choice in response.choices])
                    self.prompt_length += response.usage.prompt_tokens
                    self.completion_length += response.usage.completion_tokens
                else:
                    res.append(response.choices)
            time.sleep(self.interval)
        return res
