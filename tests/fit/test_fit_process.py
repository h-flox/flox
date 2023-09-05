import os
import pandas as pd
import pytest


from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from flox.flock import Flock
from flox.learn import federated_fit
from flox.utils.data.federate import randomly_federate_dataset

# from flox.learn.prototype import federated_fit


class MyModule(nn.Module):
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


@pytest.fixture
def data():
    return FashionMNIST(
        root=os.environ["TORCH_DATASETS"],
        download=False,
        train=True,
        transform=ToTensor(),
    )


def test_2_tier_fit(data):
    flock = Flock.from_yaml("examples/flocks/2-tier.yaml")
    fed_data = randomly_federate_dataset(
        flock,
        data,
        shuffle=True,
        random_state=None,
    )
    train_history = federated_fit(flock, MyModule, fed_data, 2)
    assert isinstance(train_history, pd.DataFrame)


def test_3_tier_fit(data):
    flock = Flock.from_yaml("examples/flocks/3-tier.yaml")
    fed_data = randomly_federate_dataset(
        flock,
        data,
        shuffle=True,
        random_state=None,
    )
    train_history = federated_fit(flock, MyModule, fed_data, 2)
    assert isinstance(train_history, pd.DataFrame)


def test_complex_fit(data):
    flock = Flock.from_yaml("examples/flocks/complex.yaml")
    fed_data = randomly_federate_dataset(
        flock,
        data,
        shuffle=True,
        random_state=None,
    )
    train_history = federated_fit(flock, MyModule, fed_data, 2)
    assert isinstance(train_history, pd.DataFrame)
