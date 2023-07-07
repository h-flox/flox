"""
This file, at the moment, is just to get an idea of how one might use FLoX one day.
In other words, this can be seen as the target quickstart demo.
"""

import argparse

import flox
import lightning as L
import os

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

from flox.aggregator import FedAvg
from flox.worker import SimpleWorkerLogic
from modules import MnistModule


class MnistAggrLogic(FedAvg):

    def on_module_evaluate(self, module: L.LightningModule):
        root = os.environ.get("PATH_DATASETS", ".")
        test_data = MNIST(root, download=True, train=True, transform=ToTensor())
        test_dataloader = DataLoader(test_data)
        trainer = L.Trainer()
        return trainer.test(module, test_dataloader)


class MnistWorkerLogic(SimpleWorkerLogic):
    def __init__(self, idx, indices):
        super().__init__(idx)
        self.name = "mnist"
        self.indices = indices

    def on_data_fetch(self):
        from torch.utils.data import Subset
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
        from os import environ

        root = environ.get("PATH_DATASETS", ".")
        data = MNIST(root, download=True, train=True, transform=ToTensor())
        data = Subset(data, indices=self.indices)
        return data

    def __len__(self) -> int:
        return len(self.indices)


def main(args: argparse.Namespace):
    workers: dict[str, MnistWorkerLogic] = flox.create_workers(30, MnistWorkerLogic)
    results = flox.federated_fit(
        global_module=MnistModule(),
        aggr=MnistAggrLogic(participation_frac=0.5),
        workers=workers,
        global_rounds=5,
        mode="local"
    )

    train_results = results["train_results"]
    test_results = results["test_results"]

    train_results.to_csv("train_results.csv")
    test_results.to_csv("test_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_workers", type=int, default=10)
    parser.add_argument("-a", "--alpha", type=float, default=100.0)
    main(parser.parse_args())