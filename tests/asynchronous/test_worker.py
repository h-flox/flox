from __future__ import annotations

import typing as t
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import pytest_asyncio

from flight.events import *
from flight.asynchronous.jobs.worker import worker_job, WorkerJobArgs
from flight.learning.module import TorchModule
from flight.strategies.strategy import DefaultStrategy

if t.TYPE_CHECKING:
    from ignite.engine import Engine
    from torch.utils.data import Dataset

@pytest.fixture
def synthetic_mnist() -> Dataset:
    """
    Generates synthetic random data (with the dimensionality of MNIST)
    for testing purposes.
    """
    g_cpu = torch.Generator()
    g_cpu.manual_seed(42)
    num_samples = 1000
    inputs = torch.randn(num_samples, 1, 28, 28, generator=g_cpu)
    targets = torch.randint(0, 10, (num_samples,), generator=g_cpu)
    return TensorDataset(inputs, targets)

class SimpleStrategy(DefaultStrategy):
    @on(WorkerEvents.STARTED)
    def on_start(self, context):
        context["on_start_ran_flag"] = True

    @on(WorkerEvents.BEFORE_TRAINING)
    def attach_metrics(self, context):
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
        print(">>> Attached metric `accuracy` to engine.")

    @on(WorkerEvents.AFTER_TRAINING)
    def print_accuracy_metric(self, context):
        trainer_state = context["trainer_state"]
        metrics = trainer_state.metrics
        print(">>> AFTER_TRAINING: ")
        print(f"    - {metrics=}")

    @on(WorkerEvents.COMPLETED)
    def on_completion(self, context):
        print("Worker job completed!")

    @on(IgniteEvents.STARTED)
    def print_hi(self, engine, context):
        context["ignite_train_start_flag"] = True

    @on(IgniteEvents.ITERATION_COMPLETED)
    def print_iteration(self, engine: Engine, context: Context):
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

    await worker_job(
        args=WorkerJobArgs(
            strategy=SimpleStrategy(),
            model=model,
            data=trainloader,
            params=model.get_params(),
            train_step=None,
            supervised=True,
        )
    )

@pytest.mark.asyncio
async def test_worker_job_with_custom_train_fn(synthetic_mnist):
    torch.manual_seed(42)
    model = SimpleModule()
    trainloader = DataLoader(synthetic_mnist, batch_size=32)

    def train_step(module, optimizer, criterion, engine: "Engine", batch):
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

    await worker_job(
        args=WorkerJobArgs(
            strategy=SimpleStrategy(),
            model=model,
            data=trainloader,
            params=model.get_params(),
            train_step=train_step,
            supervised=True,
        )
    )