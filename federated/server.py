import random
from collections import Counter
from llm.llm_api import agg_prompts

class Server:

    def __init__(self, init_prompt):

        self.global_prompt = init_prompt

    def aggregate(self, client_prompts):
        new_prompt =  agg_prompts(client_prompts)
        self.global_prompt = new_prompt
        return new_prompt