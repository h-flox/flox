from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from torchvision.datasets import MNIST

from flight.events import *
from flight.jobs.worker import worker_job, WorkerJobArgs
from flight.learning.module import TorchModule
from flight.strategy import DefaultStrategy

if t.TYPE_CHECKING:
    from ignite.engine import Engine


@pytest.fixture
def mnist() -> MNIST:
    """
    Fixture to download and return the MNIST dataset.
    """
    return MNIST(
        root="~/Research/Data/Torch-Data/",
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )


class SimpleStrategy(DefaultStrategy):
    @on(WorkerEvents.STARTED)
    def on_start(self, context):
        context["on_start_ran_flag"] = True

    @on(WorkerEvents.BEFORE_TRAINING)
    def attach_metrics(self, context):
        from ignite.metrics import Accuracy
        from ignite.engine import Engine

        def _transform(output):
            """
            NOTE: The `Accuracy()` metric requires outputs be returned
            in the order of `(y_pred, y_true)`.
            """
            y_true, y_pred = output
            return y_pred, y_true

        """
        NOTE:
        This is not working because there is an issue with the `output_transform` 
        function passed into the `create_supervised_trainer` function.
        
        The default behavior is to return a float (i.e., `loss.item()`). But, this 
        is behaving unexpectedly... see docs below:
        https://docs.pytorch.org/ignite/generated/ignite.metrics.Accuracy.html#ignite.metrics.Accuracy
        """
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


def test_worker_job(mnist):
    torch.manual_seed(42)
    model = SimpleModule().to("mps")
    trainloader = DataLoader(mnist, batch_size=32)

    worker_job(
        args=WorkerJobArgs(
            strategy=SimpleStrategy(),
            model=model,
            data=trainloader,
            params=model.get_params(),
            train_step=None,
            supervised=True,
        )
    )


def test_worker_job_with_custom_train_fn(mnist):
    torch.manual_seed(42)
    model = SimpleModule().to("cpu")
    trainloader = DataLoader(mnist, batch_size=32)

    def train_step(module, optimizer, criterion, engine: Engine, batch):
        x, y = batch
        y_pred = module(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        engine.state.metrics[engine.state.epoch] = {
            "loss": loss.item(),
        }
        # return loss.item(), y_pred, y
        return y, y_pred

    worker_job(
        args=WorkerJobArgs(
            strategy=SimpleStrategy(),
            model=model,
            data=trainloader,
            params=model.get_params(),
            train_step=train_step,
            supervised=True,
        )
    )
