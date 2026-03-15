from config import Config
from federated.trainer import FederatedTrainer
from data.dataset_loader import load_sst2, load_sst2_validation


def main():

    config = Config()

    dataset = load_sst2()[0:1000]

    validate_dataset = load_sst2_validation()

    trainer = FederatedTrainer(config, dataset,validate_dataset)

    trainer.train()


if __name__ == "__main__":

    main()