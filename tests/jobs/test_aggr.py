import pytest
import torch

from flight.jobs.aggr import aggregator_job, AggrJobArgs
from flight.state import AggregatorState, WorkerState
from flight.strategies.strategy import DefaultStrategy
from flight.learning.module import TorchModule
from flight.learning.parameters import TorchParams  
from flight.system.node import Node
from flight.jobs.protocols import Result

@pytest.fixture
def node():
    return Node(idx=1, kind="aggregator")

@pytest.fixture
def child_nodes():
    return [Node(idx=i, kind="worker") for i in range(2, 5)]

class SimpleModule(TorchModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)

    def get_params(self):
        return TorchParams({k: v.detach().clone() for k, v in self.state_dict().items()})

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def configure_criterion(self):
        return torch.nn.MSELoss()

@pytest.fixture
def child_results(child_nodes):
    results = []
    for node in child_nodes:
        state = WorkerState()
        module = SimpleModule()
        params = module.get_params()
        result = Result(
            node=node,
            state=state,
            params=params,
            module=module,
            extra={}
        )
        results.append(result)
    return results

@pytest.fixture
def strategy():
    class SyncStrategy(DefaultStrategy):
        def aggregate(self, child_params):
           
            return next(iter(child_params.values()))
    return SyncStrategy()

def test_aggregator_job_returns_result(node, child_results, strategy):
    args = AggrJobArgs(
        node=node,
        child_results=child_results,
        round_num=1,
        handlers=[],
        strategy=strategy,
    )
    result = aggregator_job(args)
    assert result.node == node
    assert isinstance(result.state, AggregatorState)
    assert hasattr(result, "params")
    assert "child_states" in result.extra
    assert "child_modules" in result.extra
    assert result.extra["round_num"] == 1

    assert result.params == strategy.aggregate({c.node.idx: c.params for c in child_results})

def test_aggregator_job_type_error_on_invalid_state(node, child_results, strategy):
    child_results[0].state = object()
    args = AggrJobArgs(
        node=node,
        child_results=child_results,
        round_num=1,
        handlers=[],
        strategy=strategy,
    )
    with pytest.raises(TypeError):
        aggregator_job(args)