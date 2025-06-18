import typing as t

import pytest
import torch.nn
from torch.optim import Optimizer, SGD

from flight.jobs.protocols import Result
from flight.learning.module import TorchModule
from flight.system import Node


@pytest.fixture
def node() -> Node:
    return Node(idx=0, kind="worker")


@pytest.fixture
def module_cls() -> t.Type[TorchModule]:
    class MyModule(TorchModule):
        def __init__(self):
            super().__init__()
            torch.manual_seed(42)
            self.layer = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.layer(x)

        def configure_optimizers(self, *args, **kwargs) -> Optimizer:
            return SGD(self.parameters(), lr=0.01)

        def configure_criterion(self, *args, **kwargs) -> t.Callable:
            return torch.nn.MSELoss()

    return MyModule


def test_result(node, module_cls):
    """"""
    result = Result(node, module=module_cls())
    assert isinstance(result, Result)
