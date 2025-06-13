import pytest
import torch
import pytest_asyncio

from flight.asynchronous.jobs.aggr import aggregator_job, AggrJobArgs
from flight.state import AggregatorState, WorkerState
from flight.strategies.strategy import DefaultStrategy
from flight.learning.module import TorchModule
from flight.learning.parameters import Params
from flight.system.node import Node
from flight.jobs.protocols import Result


class SimpleModule(TorchModule):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)

    def get_params(self):
        # Return a simple params object
        return Params({"weight": self.linear.weight.detach().clone()})

@pytest.fixture
def node():
    return Node(idx=1)

@pytest.fixture
def child_nodes():
    return [Node(idx=i) for i in range(2, 5)]

@pytest.fixture
def child_results(child_nodes):
    # Create AggregatorState or WorkerState, Params, and TorchModule for each child
    results = []
    for node in child_nodes:
        state = WorkerState()  # or AggregatorState()
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
    class AsyncStrategy(DefaultStrategy):
        async def aggregate(self, child_params):
            return next(iter(child_params.values()))
    return AsyncStrategy()

@pytest.mark.asyncio
async def test_aggregator_job_returns_result(node, child_results, strategy):
    args = AggrJobArgs(
        node=node,
        child_results=child_results,
        round_num=1,
        handlers=[],
        strategy=strategy,
    )
    result = await aggregator_job(args)
    assert result.node == node
    assert isinstance(result.state, AggregatorState)
    assert hasattr(result, "params")
    assert "child_states" in result.extra
    assert "child_modules" in result.extra
    assert result.extra["round_num"] == 1
    assert result.extra["is_async"] is True
    # Check that params match what aggregate returns
    assert result.params == await strategy.aggregate({c.node.idx: c.params for c in child_results})

@pytest.mark.asyncio
async def test_aggregator_job_type_error_on_invalid_state(node, child_results, strategy):
    # Set one child result's state to an invalid type
    child_results[0].state = object()
    args = AggrJobArgs(
        node=node,
        child_results=child_results,
        round_num=1,
        handlers=[],
        strategy=strategy,
    )
    with pytest.raises(TypeError):
        await aggregator_job(args)