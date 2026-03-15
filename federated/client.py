from prompt.prompt_optimizer import PromptOptimizer


class Client:

    def __init__(self, client_id, data):

        self.client_id = client_id

        self.data = data

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
        print(f"Client {self.client_id} best score: {best_score}")
        return local_prompt