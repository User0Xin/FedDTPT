import os

from datasets import load_dataset
import datasets



def load_sst2():

    # dataset = load_dataset("sst2",cache_dir="../dataset/sst2")
    base_path = os.path.join(os.path.dirname(__file__), "../dataset/sst2/sst2/default/0.0.0/8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb")
    dataset = load_dataset('arrow', data_files={
        'train': os.path.join(base_path, 'sst2-train.arrow'),
        'validation': os.path.join(base_path, 'sst2-validation.arrow'),
        'test': os.path.join(base_path, 'sst2-test.arrow')
    })
    data = []

    for item in dataset["train"]:

        text = item["sentence"] 
        label = item["label"]

        data.append((text,label))

    return data

if __name__ == "__main__":
    sst2_data = load_sst2()
    print(sst2_data[0])
    print(len(sst2_data))