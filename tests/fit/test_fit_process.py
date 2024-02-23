import os

import pandas as pd
import pytest
import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from flox import Flock, federated_fit
from flox.data.utils import federated_split
from flox.nn import FloxModule


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


@pytest.fixture
def data():
    return MNIST(
        root=os.environ["TORCH_DATASETS"],
        download=False,
        train=False,
        transform=ToTensor(),
    )


def test_2_tier_fit(data):
    flock = Flock.from_yaml("examples/flocks/2-tier.yaml")
    fed_data = federated_split(
        data,
        flock,
        10,
        samples_alpha=10.0,
        labels_alpha=10.0,
    )
    module, train_history = federated_fit(flock, MyModule(), fed_data, 2)
    assert isinstance(module, FloxModule)
    assert isinstance(train_history, pd.DataFrame)


def test_3_tier_fit(data):
    flock = Flock.from_yaml("examples/flocks/3-tier.yaml")
    fed_data = federated_split(
        data,
        flock,
        10,
        samples_alpha=10.0,
        labels_alpha=10.0,
    )
    module, train_history = federated_fit(flock, MyModule(), fed_data, 2)
    assert isinstance(module, FloxModule)
    assert isinstance(train_history, pd.DataFrame)


def test_complex_fit(data):
    flock = Flock.from_yaml("examples/flocks/complex.yaml")
    fed_data = federated_split(
        data,
        flock,
        10,
        samples_alpha=10.0,
        labels_alpha=10.0,
    )
    module, train_history = federated_fit(flock, MyModule(), fed_data, 2)
    assert isinstance(module, FloxModule)
    assert isinstance(train_history, pd.DataFrame)
