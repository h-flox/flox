"""
This file, at the moment, is just to get an idea of how one might use FLoX one day.
In other words, this can be seen as the target quickstart demo.
"""

import argparse

from torchvision.transforms import ToTensor

import flox
import lightning as L
import os
import random
import torch
import torch.nn.functional as F

from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from typing import Any

from flox._aggr import FedAvg
from flox._worker import WorkerLogic


class MnistAggrLogic(FedAvg):

    def __init__(self):
        super().__init__()
        self.metrics = None

    def on_module_eval(self, module: L.LightningModule):
        root = os.environ.get("PATH_DATASETS", ".")
        test_data = MNIST(root, download=True, train=True, transform=ToTensor())
        test_dataloader = torch.utils.data.DataLoader(test_data)
        trainer = L.Trainer()
        metrics = trainer.test(module, test_dataloader)
        self.metrics = metrics
        return self.metrics


class MnistWorkerLogic(WorkerLogic):
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


class MnistModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x: torch.Tensor):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def create_workers(num: int) -> dict[str, MnistWorkerLogic]:
    workers = {}
    for idx in range(num):
        n_samples = random.randint(100, 250)
        indices = random.sample(range(10_000), k=n_samples)
        workers[f"Worker-{idx}"] = MnistWorkerLogic(idx=idx, indices=list(indices))
    return workers


def main(args: argparse.Namespace):
    workers: dict[str, MnistWorkerLogic] = create_workers(10)
    results = flox.federated_fit(
        global_module=MnistModule(),
        aggr=MnistAggrLogic(),
        workers=workers,
        global_rounds=5,
        mode="local"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_workers", type=int, default=10)
    parser.add_argument("-a", "--alpha", type=float, default=100.0)
    main(parser.parse_args())
