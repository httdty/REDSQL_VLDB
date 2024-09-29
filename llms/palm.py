# -*- coding: utf-8 -*-
# @Time    : 2023/8/18 21:28
# @Author  :
# @Email   :  
# @File    : palm.py
# @Software: PyCharm

import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

from llms.llm import LLM
from llms.config import PALM_KEY
import google.generativeai as palm
from google.generativeai.types import safety_types

palm.configure(api_key=PALM_KEY)


class PaLM(LLM):
    models = {'main/text-bison-001'}

    def __init__(self, name, interval=2, **kwargs):
        super().__init__(name)
        if self.name not in self.models:
            raise LookupError("Please use valid model name for Openai model")
        self.interval = interval

    def __infer_one(self, prompt, kwargs):
        response = None
        patience = 5
        while not response:
            try:
                response = palm.generate_text(
                    model=self.name,
                    prompt=prompt,
                    max_output_tokens=128,
                    safety_settings=[
                        {
                            "category": safety_types.HarmCategory.HARM_CATEGORY_DEROGATORY,
                            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                        },
                        {
                            "category": safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE,
                            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                        },
                        {
                            "category": safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
                            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                        },
                        {
                            "category": safety_types.HarmCategory.HARM_CATEGORY_DANGEROUS,
                            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                        },{
                            "category": safety_types.HarmCategory.HARM_CATEGORY_SEXUAL,
                            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                        },
                        {
                            "category": safety_types.HarmCategory.HARM_CATEGORY_TOXICITY,
                            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                        },
                        {
                            "category": safety_types.HarmCategory.HARM_CATEGORY_UNSPECIFIED,
                            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
                        }
                    ],
                    **kwargs
                )
                if not response.result:
                    print(prompt)
                    patience -= 1
                    if patience <= 0:
                        response.candidates = [
                            {"output": "select * from table;"}
                        ]
                    else:
                        raise ValueError("Generate None Response")
            except Exception as e:
                print(e)
                time.sleep(6)
                response = None

        return response

    def infer(self, prompt_list, **kwargs):
        kwargs = {
            'stop_sequences': [kwargs['stop']],
            'candidate_count': min(kwargs['n'], 8)
        }
        assert isinstance(prompt_list, list), "Please make sure the input is a list of str"
        res = []
        with ThreadPoolExecutor(max_workers=min(len(prompt_list), 32)) as pool:
            response_list = pool.map(self.__infer_one, prompt_list, repeat(kwargs))
            for response in response_list:
                self.count += 1
                res.append([candidate['output'] for candidate in response.candidates])
            time.sleep(self.interval)
        return res
