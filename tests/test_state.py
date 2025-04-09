import typing as t

import pytest
import torch
from torch.optim import Optimizer

from flight.events import CoordinatorEvents, AggregatorEvents, WorkerEvents
from flight.learning.module import TorchModule


########################################################################################


@pytest.fixture
def params():
    class _Module(TorchModule):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.model = torch.nn.Sequential(
                torch.nn.Linear(1, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 1),
            )

        def forward(self, x):
            return self.model(x)

        def configure_optimizers(self, *args, **kwargs) -> Optimizer:
            return torch.optim.SGD(self.parameters(), lr=0.1)

        def configure_criterion(self, *args, **kwargs) -> t.Callable:
            return torch.nn.MSELoss()

    return _Module().get_params()


########################################################################################


def test_abstract_node_state():
    from flight.state import AbstractNodeState

    with pytest.raises(TypeError):
        AbstractNodeState()


def test_coord_node_state():
    from flight.state import CoordinatorState

    state = CoordinatorState(round=1)
    assert state.round == 1
    assert len(state.times) == len(CoordinatorEvents)
    assert state.seed is None


def test_aggr_node_state():
    from flight.state import AggregatorState

    state = AggregatorState()
    assert len(state.times) == len(AggregatorEvents)


def test_worker_node_state():
    from flight.state import WorkerState

    state = WorkerState()
    assert state.ignite is None
    assert len(state.times) == len(WorkerEvents)
    print(WorkerState.__bases__[0])
