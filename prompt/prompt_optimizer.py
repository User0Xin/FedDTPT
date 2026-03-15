import asyncio
import time

from data.dataset_loader import load_sst2
from llm.llm_api import query_llm,improve_prompt


class PromptOptimizer:

    def optimize(self, prompt, dataset, steps,local_prompt,best_score:float=-1.0):

        best_prompt = local_prompt
        best_score = best_score
        history = []
        improved_prompt = prompt
        for _ in range(steps):

            improved_prompt = improve_prompt(history,improved_prompt)
            # loop = asyncio.new_event_loop()
            # asyncio.set_event_loop(loop)
            # score, err_list = loop.run_until_complete(self.evaluate(improved_prompt, dataset))
            score, err_list = asyncio.run(self.evaluate(improved_prompt, dataset))
            # score, err_list = self.evaluate(improved_prompt, dataset)
            err_str = ",".join([f"sentence: {x} -> label: {y} -> llm_pred: {pred}" for x, y, pred in err_list])
            history_str = f"prompt: {improved_prompt} -> acc_score: {score} -> err_samples: [{err_str[1:100]}]"
            history.append(history_str)
            print(history_str)
            if score > best_score:
                best_score = score
                best_prompt = improved_prompt

        return best_prompt,best_score

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

        # async def predict_single(x, y):
        #     pred = await query_llm(prompt, x)
        #     return x, y, pred
        #
        # tasks = [predict_single(x, y) for x, y in dataset]
        # print("start task")
        # print(len(dataset))
        # results = await asyncio.gather(*tasks)
        # print("end task")
        # print(len(results))
        # correct = 0
        # err_list = []
        #
        # for x, y, pred in results:
        #     if pred == str(y):
        #         correct += 1
        #     else:
        #         err_list.append((x, y, pred))
        #
        # return correct / len(dataset), err_list




if __name__ == "__main__":
    optimizer = PromptOptimizer()
    prompt = "判断用户输入的句子的情绪是否为正面，是则输出1，否则输出0"
    dataset = load_sst2()[0:10]
    steps = 5
    # best_prompt = asyncio.run(optimizer.optimize(prompt, dataset,steps))
    best_prompt = optimizer.optimize(prompt, dataset, steps, None, -1.0)
    print(best_prompt)

