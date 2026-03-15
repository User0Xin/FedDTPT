from data.dataset_loader import load_sst2_validation
from prompt.prompt_optimizer import PromptOptimizer
from utils.logger import setup_logger
from utils.metrics import evaluate
# 创建 logger
logger = setup_logger("app")

class Client:

    def __init__(self, client_id, data,validation_data):

        self.client_id = client_id

        self.data = data

        self.validation_data = validation_data

        self.optimizer = PromptOptimizer()

        self.local_prompt = None
        self.best_score = -1.0

    def local_train(self, global_prompt, steps):

        local_prompt,best_score = self.optimizer.optimize(
            global_prompt,
            self.data,
            steps,
            self.local_prompt,
            self.best_score
        )
        self.local_prompt = local_prompt
        self.best_score = best_score
        logger.info(f"Client {self.client_id} best score: {best_score}, prompt: {local_prompt}")
        # print(f"Client {self.client_id} best score: {best_score}, prompt: {local_prompt}")
        score,_ = evaluate(local_prompt, self.validation_data)
        logger.info(f"Client {self.client_id} validation score: {score}")
        return local_prompt