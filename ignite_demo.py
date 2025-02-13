from __future__ import annotations

import typing as t

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from ignite.engine import Events
from torch import nn
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torchvision.transforms import ToTensor

from flight.federation.aggr import TrainJobArgs
from flight.federation.topologies import Node, WorkerState
from flight.federation.work.ignite import training_job
from flight.learning.torch import TorchModule
from flight.learning.torch.types import TensorLoss
from flight.strategies.base import DefaultWorkerStrategy

if t.TYPE_CHECKING:
    pass


def parent() -> Node:
    return Node(idx=0, kind="coordinator", children=[1])


def node() -> Node:
    return Node(idx=1, kind="worker")


def node_state() -> WorkerState:
    return WorkerState(0, None, None)


class Net(TorchModule):
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=10)
        self.model.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.RMSprop(self.parameters(), lr=0.005)

    def training_step(self, *args: t.Any, **kwargs) -> TensorLoss:
        raise NotImplementedError


def data():
    return MNIST(
        root="~/Research/Data/Torch-Data/",
        download=False,
        train=False,
        transform=ToTensor(),
    )


def get_args() -> TrainJobArgs:
    return TrainJobArgs(
        node=node(),
        parent=parent(),
        node_state=node_state(),
        model=Net().to("mps"),
        data=data(),
        worker_strategy=DefaultWorkerStrategy(),
    )


def main(args):
    result = training_job(args)
    df = pd.DataFrame.from_records(result.records)
    sns.lineplot(df, x="epoch", y="train/loss")
    plt.show()


if __name__ == "__main__":
    # def log_training_results_every_100_steps(state, trainer):

    def print_iteration(trainer, state):
        print(f"Epoch: {trainer.state.epoch}  |  Iteration: {trainer.state.iteration}")

    def print_iteration_loss(trainer, state):
        print(
            "Epoch[{}] - Iteration[{}] - Loss: {:0.5f}".format(
                trainer.state.epoch, trainer.state.iteration, trainer.state.output
            )
        )

    def log_training_results_on_epoch_end(trainer, state):
        loss, epoch = trainer.state.output, trainer.state.epoch
        record = {"train/loss": loss, "epoch": epoch}
        state["records"].append(record)
        print(f"Training Results - Epoch: {epoch}, Loss: {loss:0.5f}")

    _args = get_args()
    _args.train_handlers = [
        (Events.EPOCH_COMPLETED, log_training_results_on_epoch_end),
        # (Events.ITERATION_STARTED, print_iteration),
        (Events.ITERATION_COMPLETED(every=100), print_iteration_loss),
    ]
    main(_args)
