import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from flight.engine.data.base import BaseTransfer
from flight.federation.jobs.aggr import default_aggr_job
from flight.federation.jobs.types import Result, AggrJobArgs
from flight.federation.topologies import Node
from flight.federation.topologies.node import AggrState
from flight.learning.modules.torch import TorchModule, TorchDataModule
from flight.strategies.base import DefaultAggrStrategy


@pytest.fixture
def parent() -> Node:
    return Node(idx=0, kind="coordinator", children=[1])


@pytest.fixture
def node() -> Node:
    return Node(idx=1, kind="worker")


@pytest.fixture
def aggr_state(node) -> AggrState:
    return AggrState(0, [node], None)


@pytest.fixture
def model() -> TorchModule:
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
def result(node, aggr_state) -> Result:
    return Result(
        node=node,
        node_state=aggr_state,
        params={},
        records=[],
        extra={},
    )


@pytest.fixture
def aggr_args(node, parent, result) -> AggrJobArgs:
    return AggrJobArgs(
        node=parent,
        children=[node],
        child_results=[result],
        aggr_strategy=DefaultAggrStrategy(),
        transfer=BaseTransfer(),
    )


class TestWorkerJob:
    def test_outputs(self, aggr_args):
        result = default_aggr_job(aggr_args)
        assert isinstance(result, Result)
