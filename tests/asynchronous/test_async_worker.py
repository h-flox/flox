from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from flight.asynchronous.jobs.worker import async_worker_job, AsyncWorkerJobArgs
from flight.learning.module import TorchModule
from flight.strategies.strategy import DefaultStrategy

from ignite.engine import Engine
from torch.utils.data import Dataset

@pytest.fixture
def synthetic_mnist() -> Dataset:
    g_cpu = torch.Generator()
    g_cpu.manual_seed(42)
    num_samples = 1000
    inputs = torch.randn(num_samples, 1, 28, 28, generator=g_cpu)
    targets = torch.randint(0, 10, (num_samples,), generator=g_cpu)
    return TensorDataset(inputs, targets)

class SimpleStrategy(DefaultStrategy):
    pass  # You can add event hooks if needed, similar to your jobs/worker.py tests

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

def test_async_worker_job_supervised(synthetic_mnist):
    torch.manual_seed(42)
    model = SimpleModule()
    trainloader = DataLoader(synthetic_mnist, batch_size=32)

    results = []
    def send_result(res):
        results.append(res)

    async_worker_job(
        args=AsyncWorkerJobArgs(
            strategy=SimpleStrategy(),
            model=model,
            data=trainloader,
            params=model.get_params(),
            train_step=None,
            supervised=True,
        ),
        send_result=send_result,
    )
    # Should send a result after each epoch (default 3 epochs)
    assert len(results) == 3

def test_async_worker_job_with_custom_train_fn(synthetic_mnist):
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
        return y, y_pred

    results = []
    async_worker_job(
        args=AsyncWorkerJobArgs(
            strategy=SimpleStrategy(),
            model=model,
            data=trainloader,
            params=model.get_params(),
            train_step=train_step,
            supervised=True,
        ),
        send_result=results.append,
    )
    assert len(results) == 3

def test_async_worker_job_invalid_data():
    model = SimpleModule()
    with pytest.raises(ValueError, match="`data` must be a TorchDataModule, DataLoader, or Dataset."):
        async_worker_job(
            args=AsyncWorkerJobArgs(
                strategy=SimpleStrategy(),
                model=model,
                data=object(),  # Invalid type
                params=model.get_params(),
            ),
            send_result=lambda x: None,
        )

def test_async_worker_job_unsupervised_without_train_step(synthetic_mnist):
    model = SimpleModule()
    trainloader = DataLoader(synthetic_mnist, batch_size=32)
    with pytest.raises(ValueError, match="Unsupervised training requires a custom train_step."):
        async_worker_job(
            args=AsyncWorkerJobArgs(
                strategy=SimpleStrategy(),
                model=model,
                data=trainloader,
                params=model.get_params(),
                supervised=False,
            ),
            send_result=lambda x: None,
        )