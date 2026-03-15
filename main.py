from config import Config
from federated.trainer import FederatedTrainer
from data.dataset_loader import load_sst2



def main():

    config = Config()

    dataset = load_sst2()[0:300]


    trainer = FederatedTrainer(config, dataset)

    trainer.train()


if __name__ == "__main__":

    main()