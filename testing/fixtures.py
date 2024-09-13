import numpy as np
import pytest
import torch
from lightning import LightningDataModule, LightningModule
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from flight.engine.control.serial import SerialController
from flight.federation.topologies import Node
from flight.federation.topologies.node import NodeKind, WorkerState
from flight.learning.scikit import ScikitDataModule
from flight.learning.torch import TorchModule

_SEED = 42
"""
Random seed used for random number generation.
"""


@pytest.fixture
def lightning_data_module() -> LightningDataModule:
    class SimpleLightningDataModule(LightningDataModule):
        def __init__(self):
            super().__init__()

            self.num_samples = 10_000
            self.num_features = 1

            g = torch.Generator(device="cpu")
            g.manual_seed(0)

            x = torch.randn((self.num_samples, self.num_features), generator=g)
            y = torch.randn((self.num_samples, 1), generator=g)
            self.raw_data = TensorDataset(x, y)

        def train_dataloader(self) -> DataLoader:
            subset = Subset(self.raw_data, indices=list(range(0, 8_000)))
            return DataLoader(subset, batch_size=32)

        def val_dataloader(self) -> DataLoader | None:
            subset = Subset(self.raw_data, indices=list(range(8_000, 9_000)))
            return DataLoader(subset, batch_size=32)

        def test_dataloader(self) -> DataLoader | None:
            subset = Subset(self.raw_data, indices=list(range(9_000, 10_000)))
            return DataLoader(subset, batch_size=32)

    return SimpleLightningDataModule()


@pytest.fixture
def serial_controller() -> SerialController:
    return SerialController()


@pytest.fixture
def worker_node() -> Node:
    node = Node(idx=0, kind=NodeKind.WORKER)
    return node


@pytest.fixture
def lightning_module() -> LightningModule:
    class SimpleLightningModule(LightningModule):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(1, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
            )

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_nb):
            inputs, targets = batch
            preds = self(inputs)
            loss = F.l1_loss(preds, targets)
            self.logger.log_metrics({"loss": loss.item()})
            return loss

        def validation_step(self, batch, batch_nb):
            return self.training_step(batch, batch_nb)

        def test_step(self, batch, batch_nb):
            inputs, labels = batch
            prediction = self(inputs)
            loss = F.l1_loss(prediction, labels)

            _, prediction = torch.max(prediction, 1)

            return loss

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.SGD(self.parameters(), lr=0.001)

    return SimpleLightningModule()


@pytest.fixture
def scikit_regr_data_module() -> ScikitDataModule:
    class MySciKitDataModule(ScikitDataModule):
        def __init__(self) -> None:
            super().__init__()
            self.num_samples = 10_000
            self.num_features = 1

            self.x = np.random.randn(self.num_samples, self.num_features)
            self.y = np.random.randn(self.num_samples)

            self.x_train, self.x_temp, self.y_train, self.y_temp = train_test_split(
                self.x, self.y, test_size=0.4, random_state=42
            )
            self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(
                self.x_temp, self.y_temp, test_size=0.5, random_state=42
            )

        def train_data(self, node: Node | None = None):
            return self.x_train, self.y_train

        def test_data(self, node: Node | None = None):
            return self.x_test, self.y_test

        def valid_data(self, node: Node | None = None):
            return self.x_val, self.y_val

    return MySciKitDataModule()


@pytest.fixture
def scikit_clf_data_module() -> ScikitDataModule:
    class MyClassificationDataModule(ScikitDataModule):
        def __init__(self):
            inputs, labels = make_classification(
                n_samples=10000, n_features=20, random_state=_SEED
            )

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                inputs, labels, test_size=0.3
            )

        def train_data(self, node: Node | None = None):
            return self.x_train, self.y_train

        def test_data(self, node: Node | None = None):
            return self.x_test, self.y_test

        def valid_data(self, node: Node | None = None):
            return self.x_test, self.y_test

    return MyClassificationDataModule()


@pytest.fixture
def valid_module() -> TorchModule:
    class TestModule(TorchModule):
        def __init__(self):
            super().__init__()
            torch.manual_seed(_SEED)
            self.m = torch.nn.Parameter(torch.tensor([1.0]))
            self.b = torch.nn.Parameter(torch.tensor([3.0]))

        def forward(self, x):
            return self.m * x + self.b

        def training_step(self, batch, batch_nb):
            return self(batch)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    return TestModule()


@pytest.fixture
def invalid_module_cls() -> type[TorchModule]:
    class TestModule(TorchModule):  # noqa
        def __init__(self):
            super().__init__()
            torch.manual_seed(_SEED)
            self.m = torch.nn.Parameter(torch.tensor([1.0]))
            self.b = torch.nn.Parameter(torch.tensor([3.0]))

        def forward(self, x):
            return self.m * x + self.b

    return TestModule


@pytest.fixture
def worker_state() -> WorkerState:
    return WorkerState(0, None, None)
