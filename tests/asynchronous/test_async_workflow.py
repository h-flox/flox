import torch
from torch.utils.data import TensorDataset

from flight.asynchronous.workflow import AsyncWorkflow, WorkerTimeTracker
from flight.learning.module import TorchModule
from flight.runtime import Runtime
from flight.strategies.strategy import DefaultStrategy
from flight.system.utils import flat_topology


class SimpleModule(TorchModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)

    def configure_criterion(self, *args, **kwargs):
        return torch.nn.CrossEntropyLoss()

    def configure_optimizers(self, *args, **kwargs):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_asyncworkflow_init():
    module = SimpleModule()
    dataset = TensorDataset(torch.randn(20, 4), torch.randint(0, 2, (20,)))
    topology = flat_topology(2)
    runtime = Runtime.simple_setup(max_workers=2)
    wf = AsyncWorkflow(
        runtime,
        topology,
        1,
        module,
        dataset,
        strategy=DefaultStrategy(),
    )
    assert wf.module is module
    assert wf.dataset is dataset
    assert wf.topology is topology
    assert wf.runtime is runtime


def test_asyncworkflow_run_minimal():
    module = SimpleModule()
    dataset = TensorDataset(torch.randn(20, 4), torch.randint(0, 2, (20,)))
    topology = flat_topology(2)
    runtime = Runtime.simple_setup(max_workers=2)
    wf = AsyncWorkflow(
        runtime,
        topology,
        1,
        module,
        dataset,
        strategy=DefaultStrategy(),
    )
    model, _ = wf.start()
    assert isinstance(model, SimpleModule)


def test_asyncworkflow_custom_loss():
    def custom_loss(model, data_loader, device=None):
        return 42.0

    module = SimpleModule()
    dataset = TensorDataset(torch.randn(20, 4), torch.randint(0, 2, (20,)))
    topology = flat_topology(2)
    runtime = Runtime.simple_setup(max_workers=2)
    wf = AsyncWorkflow(
        runtime,
        topology,
        1,
        module,
        dataset,
        strategy=DefaultStrategy(),
        loss_function=custom_loss,
    )
    assert wf.loss_function is custom_loss
    # Optionally call the loss function
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    assert wf.loss_function(module, loader) == 42.0


def test_asyncworkflow_worker_time_tracker():
    module = SimpleModule()
    dataset = TensorDataset(torch.randn(20, 4), torch.randint(0, 2, (20,)))
    topology = flat_topology(2)
    runtime = Runtime.simple_setup(max_workers=2)
    tracker = WorkerTimeTracker(num_workers=2)
    wf = AsyncWorkflow(
        runtime,
        topology,
        1,
        module,
        dataset,
        strategy=DefaultStrategy(),
        worker_time_tracker=tracker,
    )
    wf.start()
    # At least one job time should be recorded for each worker
    for times in tracker.job_times.values():
        assert isinstance(times, list)
