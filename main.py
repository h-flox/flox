import os

import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from flox import Topology, federated_fit
from flox.learn import FloxModule
from flox.learn.data.utils import federated_split


class MyModule(FloxModule):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.last_accuracy = torch.tensor([1.0])

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self(inputs)
        loss = nn.functional.cross_entropy(preds, targets)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=1e-3)


def load_data():
    return MNIST(
        root=os.environ["TORCH_DATASETS"],
        download=False,
        train=False,
        transform=ToTensor(),
    )


if __name__ == "__main__":
    flock = Topology.from_yaml("examples/flocks/3-tier.yaml")
    fed_data = federated_split(
        load_data(),
        flock,
        10,
        samples_alpha=10.0,
        labels_alpha=10.0,
    )
    module, train_history = federated_fit(
        flock, MyModule(), fed_data, 2, strategy="fedavg", launcher_kind="thread"
    )
