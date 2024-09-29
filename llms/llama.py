# -*- coding: utf-8 -*-
# @Time    : 2023/4/10 09:13
# @Author  :
# @Email   :  
# @File    : llama.py
# @Software: PyCharm
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from typing import List

from openai import OpenAI

from llms.config import LLAMA_KEY
from llms.llm import LLM



class LLAMA(LLM):
    models = {'meta-llama/Meta-Llama-3-70B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct', 'Qwen/Qwen2-72B-Instruct'}

    def __init__(self, name, interval=0, **kwargs):
        super().__init__(name)
        api_key = LLAMA_KEY
        if 'api_key' in kwargs:
            if kwargs['api_key']:
                api_key = kwargs['api_key']
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")
        self.api = self.client.chat.completions.create
        self.interval = interval

    def __infer_one(self, prompt, kwargs):
        response = None
        if not prompt:
            return response
        self.count += 1
        if 'n' in kwargs and kwargs['n'] > 1:
            kwargs['n'] = 1
            print("LLAMA only support n=1")
        while not response:
            try:
                response = self.api(
                    model=self.name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    # temperature=0,
                    # max_tokens=128,
                    # top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    timeout=120,
                    **kwargs
                )
            except Exception as e:
                print(e)
                time.sleep(12)
                response = None

        return response

    def infer(self, prompt_list: List[str], **kwargs):
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
