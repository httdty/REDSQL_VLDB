# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 10:25
# @Author  :
# @Email   :  
# @File    : moss.py
# @Software: PyCharm
from typing import List

import requests

from llms.config import MOSS_KEY
from llms.llm import LLM


class Moss(LLM):
    def __init__(self, name, interval=2, **kwargs):
        super().__init__(name)
        self.api_key = MOSS_KEY
        self.interval = interval
        # self.api_url = "http/inference"  # In-school
        self.api_url = "http:/inference"  # Out-school
        self.headers = {
            "apikey": self.api_key
        }

    def infer(self, prompt_list: List[str], context=None):
        res = []
        for prompt in prompt_list:
            data = {
                "request": prompt
            }

            if context:
                data["context"] = context

            response = requests.post(self.api_url, headers=self.headers, json=data)
            try:
                res.append(response.json()['response'])
            except KeyError:
                print(response.json())
                res.append("")
        return res

# if __name__ == "__main__":
#     moss = Moss("MOSS")
#
#     request_text = "hi"
#     response = moss.infer(request_text)
#     print(json.dumps(response, indent=2, ensure_ascii=False))
#
#     context_text = response["context"]
#     request_text = "what's your name?"
#     response = moss.infer(request_text, context=context_text)
#     print(json.dumps(response, indent=2, ensure_ascii=False))
