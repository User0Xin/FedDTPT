import asyncio

from llm.llm_api import query_llm


def accuracy(preds, labels):

    correct = 0

    for p,l in zip(preds,labels):

        if p == l:
            correct += 1

    return correct/len(labels)

async def evaluate(prompt, dataset):
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