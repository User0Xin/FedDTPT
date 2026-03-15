import asyncio

from data.dataset_loader import load_sst2_validation
from federated.client import Client
from federated.server import Server
from llm.llm_api import query_llm
from prompt.prompt_utils import init_prompt
from utils.logger import setup_logger
from utils.metrics import evaluate
# 创建 logger
logger = setup_logger("app")

class FederatedTrainer:

    def __init__(self, config, dataset,valid_dataset):

        self.config = config

        self.dataset = dataset

        self.valid_dataset = valid_dataset

        self.clients = []

        self.init_clients()

        init_p = init_prompt()

        self.server = Server(init_p)

    def init_clients(self):

        split = len(self.dataset) // self.config.num_clients

        for i in range(self.config.num_clients):

            data = self.dataset[i*split:(i+1)*split]

            self.clients.append(Client(i, data,self.valid_dataset))

    def train(self):

        for r in range(self.config.num_rounds):

            logger.info("Round %d", r)

            global_prompt = self.server.global_prompt

            client_prompts = []

            for client in self.clients:

                local_prompt = client.local_train(
                    global_prompt,
                    self.config.local_steps
                )

                client_prompts.append(local_prompt)

            new_prompt = self.server.aggregate(client_prompts)

            # print("Global Prompt:", " ".join(new_prompt))
            logger.info("Global Prompt: %s", " ".join(new_prompt))

            score, err_list = asyncio.run(evaluate(new_prompt, self.valid_dataset))
            logger.info("Accuracy: %f", score)
            # print("Accuracy:", score)

