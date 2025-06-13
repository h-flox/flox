import pytest
import torch
import pytest_asyncio
from torch.utils.data import DataLoader, TensorDataset

from flight.events import *
from flight.asynchronous.jobs.worker import worker_job, WorkerJobArgs
from flight.learning.module import TorchModule
from flight.strategies.strategy import DefaultStrategy
from flight.learning.parameters import Params
from flight.system.node import Node

if torch.__version__ < "2":
    torch.set_default_dtype(torch.float32)

if t.TYPE_CHECKING:
    from ignite.engine import Engine
    from torch.utils.data import Dataset

@pytest.fixture
def synthetic_mnist():
    g_cpu = torch.Generator()
    g_cpu.manual_seed(42)
    num_samples = 1000
    inputs = torch.randn(num_samples, 1, 28, 28, generator=g_cpu)
    targets = torch.randint(0, 10, (num_samples,), generator=g_cpu)
    return TensorDataset(inputs, targets)

class SimpleStrategy(DefaultStrategy):
    @on(WorkerEvents.STARTED)
    async def on_start(self, context):
        context["on_start_ran_flag"] = True

    @on(WorkerEvents.BEFORE_TRAINING)
    async def attach_metrics(self, context):
        from ignite.metrics import Accuracy
        from ignite.engine import Engine

        def _transform(output):
            y_true, y_pred = output
            return y_pred, y_true

        engine = context["trainer"]
        if not isinstance(engine, Engine):
            raise TypeError("Expected an instance of `Engine`.")

        metric = Accuracy(output_transform=_transform)
        metric.attach(engine, "accuracy")

    @on(WorkerEvents.AFTER_TRAINING)
    async def print_accuracy_metric(self, context):
        trainer_state = context["trainer_state"]
        metrics = trainer_state.metrics
        print(">>> AFTER_TRAINING: ")
        print(f"    - {metrics=}")

    @on(WorkerEvents.COMPLETED)
    async def on_completion(self, context):
        print("Worker job completed!")

    @on(IgniteEvents.STARTED)
    async def print_hi(self, engine, context):
        context["ignite_train_start_flag"] = True

    @on(IgniteEvents.ITERATION_COMPLETED)
    async def print_iteration(self, engine, context):
        print(f">>> Iteration {engine.state.iteration} completed!")
        print(f"    - {engine.state.metrics=}")
        print("---")

class SimpleModule(TorchModule):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def configure_criterion(self):
        return torch.nn.CrossEntropyLoss()

@pytest.mark.asyncio
async def test_worker_job(synthetic_mnist):
    torch.manual_seed(42)
    model = SimpleModule()
    trainloader = DataLoader(synthetic_mnist, batch_size=32)

    result = await worker_job(
        args=WorkerJobArgs(
            strategy=SimpleStrategy(),
            model=model,
            data=trainloader,
            params=model.get_params(),
            train_step=None,
            supervised=True,
        )
    )
    assert result is not None
    assert hasattr(result, "params")
    assert hasattr(result, "module")

@pytest.mark.asyncio
async def test_worker_job_with_custom_train_fn(synthetic_mnist):
    torch.manual_seed(42)
    model = SimpleModule()
    trainloader = DataLoader(synthetic_mnist, batch_size=32)

    def train_step(module, optimizer, criterion, engine, batch):
        x, y = batch
        y_pred = module(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        engine.state.metrics[engine.state.epoch] = {
            "loss": loss.item(),
        }
        return y, y_pred

    result = await worker_job(
        args=WorkerJobArgs(
            strategy=SimpleStrategy(),
            model=model,
            data=trainloader,
            params=model.get_params(),
            train_step=train_step,
            supervised=True,
        )
    )
    assert result is not None
    assert hasattr(result, "params")
    assert hasattr(result, "module")