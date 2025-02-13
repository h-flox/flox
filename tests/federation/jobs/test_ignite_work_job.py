import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from flight.federation.jobs.types import TrainJobArgs, Result
from flight.federation.jobs.work_ignite import training_job
from flight.federation.topologies import Node
from flight.federation.topologies.node import WorkerState
from flight.learning.torch import TorchDataModule
from flight.learning.torch import TorchModule
from flight.strategies.base import DefaultWorkerStrategy, DefaultTrainerStrategy


@pytest.fixture
def parent() -> Node:
    return Node(idx=0, kind="coordinator", children=[1])


@pytest.fixture
def node() -> Node:
    return Node(idx=1, kind="worker")


@pytest.fixture
def node_state() -> WorkerState:
    return WorkerState(0, None, None)


@pytest.fixture
def linear_regr_model() -> TorchModule:
    class LinearRegr(TorchModule):
        def __init__(self):
            super().__init__()
            self.m = torch.nn.Parameter(
                torch.randn(
                    1,
                )
            )
            self.b = torch.nn.Parameter(
                torch.randn(
                    1,
                )
            )

        def forward(self, x):
            return self.m * x + self.b

        def training_step(self, batch, batch_idx):
            inputs, targets = batch
            preds = self(inputs)
            loss = torch.nn.functional.l1_loss(preds, targets)
            return loss

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.SGD(self.parameters(), lr=0.001)

        def configure_criterion(self) -> nn.Module:
            return nn.L1Loss()

    return LinearRegr()


@pytest.fixture
def data() -> TorchDataModule:
    class DataModule(TorchDataModule):
        def __init__(self):
            super().__init__()

            def fn(xi):
                return 12.34 * xi + 56.789

            self.num_items = 100
            x = torch.tensor([[val] for val in np.linspace(0.0, 10.0, self.num_items)])
            y = torch.tensor([[fn(xi)] for xi in x])
            self.data = TensorDataset(x, y)

        def train_data(self, node: Node | None = None) -> DataLoader:
            start = 0
            end = int(self.num_items * 0.8)
            subset = Subset(self.data, indices=list(range(start, end)))
            return DataLoader(subset, batch_size=16)

        def valid_data(self, node: Node | None = None) -> DataLoader | None:
            return None

        def test_data(self, node: Node | None = None) -> DataLoader | None:
            return None

    return DataModule()


@pytest.fixture
def train_args(node, parent, node_state, linear_regr_model, data) -> TrainJobArgs:
    return TrainJobArgs(
        node,
        parent,
        node_state,
        linear_regr_model,
        data,
        worker_strategy=DefaultWorkerStrategy(),
        trainer_strategy_depr=DefaultTrainerStrategy(),
    )


class TestWorkerJob:
    def test_outputs(self, train_args):
        result = training_job(train_args)
        assert isinstance(result, Result)
