import random


class PromptMutator:

    def __init__(self, vocab):
        self.vocab = vocab

    def replace(self, prompt):

        new_prompt = prompt.copy()

        idx = random.randint(0, len(prompt)-1)

        new_prompt[idx] = random.choice(self.vocab)

        return new_prompt

    def swap(self, prompt):

        new_prompt = prompt.copy()

        i, j = random.sample(range(len(prompt)), 2)

        new_prompt[i], new_prompt[j] = new_prompt[j], new_prompt[i]

        return new_prompt

    def insert(self, prompt):

        new_prompt = prompt.copy()

        idx = random.randint(0, len(prompt))

        new_prompt.insert(idx, random.choice(self.vocab))

        return new_prompt[:len(prompt)]

    def mutate(self, prompt):

        op = random.choice(["replace","swap","insert"])

        if op == "replace":
            return self.replace(prompt)

        if op == "swap":
            return self.swap(prompt)

        return self.insert(prompt)