import unittest
from copy import deepcopy

import torch
from torch.utils.data import TensorDataset

from flight.asynchronous.workflow import AsyncStrategy, AsyncStrategyState, AsyncStrategyEvents
from flight.system.node import Node, NodeKind
from flight.system.topology import Topology
from flight.runtime import Runtime
from flight.strategies.strategy import DefaultStrategy
from flight.learning.module import TorchModule

class MinimalTorchModule(TorchModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.include_state = False
    def forward(self, x):
        return self.linear(x)
    def configure_criterion(self, *args, **kwargs):
        return torch.nn.MSELoss()
    def configure_optimizers(self, *args, **kwargs):
        return torch.optim.SGD(self.parameters(), lr=0.01)

class TestAsyncStrategy(unittest.TestCase):
    def setUp(self):
        self.num_workers = 3
        self.runtime = Runtime.simple_setup(max_workers=2, exec_kind="thread")
        self.nodes = [Node(idx=i, kind=NodeKind.WORKER) for i in range(self.num_workers)]
        self.edges = [(i, (i+1)%self.num_workers) for i in range(self.num_workers)]  # simple ring
        self.topology = Topology(self.nodes, self.edges)
        self.num_global_rounds = 2
        self.module = MinimalTorchModule()
        self.dataset = TensorDataset(torch.randn(10, 2), torch.randint(0, 2, (10, 1)))
        self.strategy = DefaultStrategy()

    def test_async_strategy_initialization(self):
        async_strategy = AsyncStrategy(
            runtime=self.runtime,
            topology=self.topology,
            num_global_rounds=self.num_global_rounds,
            module=self.module,
            dataset=self.dataset,
            strategy=self.strategy,
        )
        # Check state initialization
        self.assertIsInstance(async_strategy.state, AsyncStrategyState)
        self.assertTrue(all(idx in async_strategy.state.worker_rounds for idx in range(self.num_workers)))
        self.assertTrue(all(idx in async_strategy.state.worker_params for idx in range(self.num_workers)))
        self.assertEqual(async_strategy.state.completed_worker_jobs, 0)
        for idx in range(self.num_workers):
            worker_params = async_strategy.state.worker_params[idx]
            module_params = self.module.get_params()
            # Ensure both are torch tensors for comparison
            worker_params_torch = worker_params.torch() if hasattr(worker_params, 'torch') else worker_params
            module_params_torch = module_params.torch() if hasattr(module_params, 'torch') else module_params
            for key in module_params_torch:
                self.assertTrue(
                    torch.allclose(worker_params_torch[key], module_params_torch[key]),
                    f"Parameter '{key}' for worker {idx} does not match."
                )
            self.assertEqual(async_strategy.state.worker_rounds[idx], 0)

    def test_async_strategy_start(self):
        async_strategy = AsyncStrategy(
            runtime=self.runtime,
            topology=self.topology,
            num_global_rounds=self.num_global_rounds,
            module=self.module,
            dataset=self.dataset,
            strategy=self.strategy,
        )
        # Since start is not fully implemented, just check it runs and returns a tuple
        result = async_strategy.start()
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], async_strategy.module)

    def test_async_strategy_events_enum(self):
        self.assertEqual(AsyncStrategyEvents.STARTED.value, "started")
        self.assertEqual(AsyncStrategyEvents.COMPLETED.value, "completed")
        self.assertEqual(AsyncStrategyEvents.AGGREGATION_COMPLETED.value, "aggregation_completed")
        self.assertEqual(AsyncStrategyEvents.WORKER_JOB_STARTED.value, "worker_job_started")
        self.assertEqual(AsyncStrategyEvents.WORKER_JOB_COMPLETED.value, "worker_job_completed")


