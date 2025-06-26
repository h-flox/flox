from flight.runtime import Runtime
from flight.asynchronous.jobs.strategy import AsyncStrategy
from flight.learning.module import TorchModule
from flight.system.utils import flat_topology
from torch.utils.data import TensorDataset
from flight.strategies.strategy import DefaultStrategy, Strategy
import typing as t
import torch

class DefaultAsyncStrategy(AsyncStrategy):
    def __init__(
        self,
        runtime: Runtime,
        topology: t.Any,
        num_global_rounds: int,
        module: TorchModule,
        dataset: TensorDataset,
        strategy: Strategy,
    ):
        super().__init__(
            runtime=runtime,
            topology=topology,
            num_global_rounds=num_global_rounds,
            module=module,
            dataset=dataset,
            strategy=strategy,
        )

class AsyncWorkflow:

    def __init__(
        self,
        module: TorchModule,
        topo: t.Any,
        strategy: t.Optional[AsyncStrategy] = None,
        dataset: t.Optional[TensorDataset] = None,
        runtime: t.Optional[Runtime] = None,
        num_global_rounds: int = 1,
    ):
        self.module = module
        self.topo = topo
        self.runtime = runtime or Runtime.simple_setup(max_workers=len(topo.workers))
        self.dataset = dataset or self.default_dataset()
        self.num_global_rounds = num_global_rounds
        if strategy is None:
            self.strategy = DefaultAsyncStrategy(
                runtime=self.runtime,
                topology=self.topo,
                num_global_rounds=self.num_global_rounds,
                module=self.module,
                dataset=self.dataset,
                strategy=DefaultStrategy(),
            )
        else:
            self.strategy = strategy

    def default_dataset(self):
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        return TensorDataset(X, y)

    def start(self):
        """Start the asynchronous federated learning workflow."""
        return self.strategy.start()

class MyModule(TorchModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

    def configure_criterion(self, *args, **kwargs):
        return torch.nn.CrossEntropyLoss()

    def configure_optimizers(self, *args, **kwargs):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.linear(x)
        loss = self.configure_criterion()(logits, y)
        return loss

if __name__ == "__main__":
    module = MyModule()
    dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    topology = flat_topology(10)
    runtime = Runtime.simple_setup(max_workers=len(topology.workers))
    num_global_rounds = 1

    strategy = DefaultAsyncStrategy(
        runtime=runtime,
        topology=topology,
        num_global_rounds=num_global_rounds,
        module=module,
        dataset=dataset,
        strategy=DefaultStrategy(),
    )
    wf = AsyncWorkflow(
        module=module,
        topo=topology,
        strategy=strategy,
        dataset=dataset,
        runtime=runtime,
        num_global_rounds=num_global_rounds,
    )
    wf.start()
