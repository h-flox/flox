"""
This file, at the moment, is just to get an idea of how one might use FLoX one day.
In other words, this can be seen as the target quickstart demo.
"""

import argparse
import lightning as L
import os
import torch
import torch.nn.functional as F

from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from typing import Any


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


def main(args: argparse.Namespace):
    # import flox
    module = MnistModule()
    dataset = MNIST(
        os.environ.get("PATH_DATASETS", "."),
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    workers: dict[WorkerID, WorkerLogic] = flox.worker.WorkerModule.federate_data(
        dataset,
        num_workers=args.num_workers,
        alpha=args.alpha
    )
    module, metrics = flox.federated_fit(
        workers,
        aggr="fedavg",
        mode="local"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_workers", type=int, default=10)
    parser.add_argument("-a", "--alpha", type=float, default=100.0)
    main(parser.parse_args())
