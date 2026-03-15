import asyncio

from federated.client import Client
from federated.server import Server
from llm.llm_api import query_llm
from prompt.prompt_utils import init_prompt


class FederatedTrainer:

    def __init__(self, config, dataset):

        self.config = config

        self.dataset = dataset

        self.clients = []

        self.init_clients()

        init_p = init_prompt()

        self.server = Server(init_p)

    def init_clients(self):

        split = len(self.dataset) // self.config.num_clients

        for i in range(self.config.num_clients):

            data = self.dataset[i*split:(i+1)*split]

            self.clients.append(Client(i, data))

    def train(self):

        for r in range(self.config.num_rounds):

            print("Round", r)

            global_prompt = self.server.global_prompt

            client_prompts = []

            for client in self.clients:

                local_prompt = client.local_train(
                    global_prompt,
                    self.config.local_steps
                )

                client_prompts.append(local_prompt)

            new_prompt = self.server.aggregate(client_prompts)

            print("Global Prompt:", " ".join(new_prompt))

            score, err_list = asyncio.run(self.evaluate(new_prompt, self.dataset))
            print("Accuracy:", score)

    async def evaluate(self, prompt, dataset):
        async def predict_batch(batch_data):
            correct = 0
            err_list = []
            for x, y in batch_data:
                pred = await query_llm(prompt, x)
                if pred == str(y):
                    correct += 1
                else:
                    err_list.append((x, y, pred))
            return correct, err_list

        batch_size = max(1, len(dataset) // 100)
        batches = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            batches.append(batch)

        tasks = [predict_batch(batch) for batch in batches]
        print("start task")
        results = await asyncio.gather(*tasks)
        print("end task")

        total_correct = 0
        all_err_list = []

        for correct, err_list in results:
            total_correct += correct
            all_err_list.extend(err_list)

        return total_correct / len(dataset), all_err_list