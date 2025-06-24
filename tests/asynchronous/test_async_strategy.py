import pytest
import torch
from torch.utils.data import TensorDataset

from flight.asynchronous.jobs.strategy import AsyncStrategy, AsyncStrategyEvents
from flight.system.utils import flat_topology
from flight.learning.parameters import parameters, NumpyParams
from flight.jobs.protocols import Result
from flight.runtime import Runtime
import concurrent.futures

import torch.nn as nn
from flight.learning.module import TorchModule

class SimpleModule(TorchModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    def forward(self, x):
        return self.linear(x)
    def configure_criterion(self, *args, **kwargs):
        return torch.nn.CrossEntropyLoss()
    def configure_optimizers(self, *args, **kwargs):
        return torch.optim.SGD(self.parameters(), lr=0.01)

class SimpleStrategy:
    def __init__(self):
        self.event_calls = []
    def fire_event_handler(self, event_type, context=None, when=None):
        self.event_calls.append((event_type, context))
    def get_event_handlers_by_genre(self, *args, **kwargs):
        return []

@pytest.fixture
def simple_topology():
    return flat_topology(2)

@pytest.fixture
def simple_module():
    return SimpleModule()

@pytest.fixture
def simple_dataset():
    X = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    return TensorDataset(X, y)

@pytest.fixture
def simple_strategy():
    return SimpleStrategy()

@pytest.fixture
def simple_runtime():
    class SimpleRuntime(Runtime):
        def __init__(self):
            pass
        def submit(self, fn, args):
            f = concurrent.futures.Future()
            params = args.model.get_params()
            f.set_result(Result(node=args.node, module=args.model, params=params))
            return f
    return SimpleRuntime()

def test_async_strategy_init(simple_module, simple_topology, simple_dataset, simple_runtime, simple_strategy):
    strat = AsyncStrategy(
        runtime=simple_runtime,
        topology=simple_topology,
        num_global_rounds=1,
        module=simple_module,
        dataset=simple_dataset,
        strategy=simple_strategy,
    )
    assert strat.runtime is simple_runtime
    assert strat.topology is simple_topology
    assert strat.module is simple_module
    assert strat.dataset is simple_dataset
    assert strat.strategy is simple_strategy
    assert isinstance(strat.state.global_params, NumpyParams)
    assert set(strat.state.worker_params.keys()) == set([n.idx for n in simple_topology.workers])
    assert set(strat.state.worker_rounds.keys()) == set([n.idx for n in simple_topology.workers])
    assert strat.state.completed_worker_jobs == 0

def test_add_and_fire_event_handler(simple_module, simple_topology, simple_dataset, simple_runtime, simple_strategy):
    strat = AsyncStrategy(
        runtime=simple_runtime,
        topology=simple_topology,
        num_global_rounds=1,
        module=simple_module,
        dataset=simple_dataset,
        strategy=simple_strategy,
    )
    called = {}
    def handler(context):
        called['fired'] = context
    strat.fire_event_handler = lambda event_type, context=None: handler(context)
    strat.add_event_handler(AsyncStrategyEvents.STARTED, handler)
    strat.fire_event_handler(AsyncStrategyEvents.STARTED, {'foo': 'bar'})
    assert 'fired' in called
    assert called['fired'] == {'foo': 'bar'}

def test_partial_aggregation_policy(simple_module, simple_topology, simple_dataset, simple_runtime, simple_strategy):
    strat = AsyncStrategy(
        runtime=simple_runtime,
        topology=simple_topology,
        num_global_rounds=1,
        module=simple_module,
        dataset=simple_dataset,
        strategy=simple_strategy,
    )
    p1 = parameters({'weight': torch.ones(2, 2)}, backend='torch')
    p2 = parameters({'weight': torch.zeros(2, 2)}, backend='torch')
    worker_ids = [n.idx for n in simple_topology.workers]
    strat.state.worker_params = {worker_ids[0]: p1, worker_ids[1]: p2}
    strat.partial_aggregation_policy()

    avg = strat.state.global_params['weight']
    assert torch.allclose(avg, torch.ones(2, 2) * 0.5)

def test_start_runs(simple_module, simple_topology, simple_dataset, simple_strategy):
    class SimpleRuntime(Runtime):
        def __init__(self):
            pass
        def submit(self, fn, args):
            f = concurrent.futures.Future()
            params = args.model.get_params()
            f.set_result(Result(node=args.node, module=args.model, params=params))
            return f
    runtime = SimpleRuntime()
    strat = AsyncStrategy(
        runtime=runtime,
        topology=simple_topology,
        num_global_rounds=1,
        module=simple_module,
        dataset=simple_dataset,
        strategy=simple_strategy,
    )
    events = []

    strat.fire_event_handler = lambda event_type, context=None: events.append((event_type, context))
    module, history = strat.start()
    assert isinstance(module, SimpleModule)
    assert history is None

    event_types = [e[0] for e in events]
    assert AsyncStrategyEvents.STARTED in event_types
    assert AsyncStrategyEvents.WORKER_JOB_STARTED in event_types
    assert AsyncStrategyEvents.WORKER_JOB_COMPLETED in event_types
    assert AsyncStrategyEvents.AGGREGATION_COMPLETED in event_types
    assert AsyncStrategyEvents.COMPLETED in event_types
