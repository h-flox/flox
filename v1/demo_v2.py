import typing as t

import torch
from ignite.engine import Events

from torch.utils.data import TensorDataset, DataLoader

from v1.flight.federation.work.ignite import training_job
from v1.flight import TorchModule, TorchDataModule
from v1.flight import TensorLoss
from v1.flight import Node


####################################################################################################


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


####################################################################################################


class MyModule(TorchModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.01)

    def configure_criterion(self):
        return torch.nn.MSELoss()

    def training_step(self, *args: t.Any, **kwargs) -> TensorLoss:
        pass


class MyData(TorchDataModule):
    def __init__(self, n: int, seed: int = 42):
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.data = TensorDataset(
            torch.randn(n, 1, generator=generator),
            torch.randn(n, 1, generator=generator),
        )

    def train_data(self, node: Node | None = None):
        return DataLoader(self.data)


def local_training_no_topology():
    model = MyModule().to("mps")
    data = MyData(1000)
    training_job(
        model,
        data,
        train_handlers=[
            (Events.EPOCH_COMPLETED, log_training_results_on_epoch_end),
            (Events.ITERATION_COMPLETED(every=100), print_iteration_loss),
        ],
    )


def main():
    model = MyModule()

    # fed = SyncFederationV2()
    # fed.start()


if __name__ == "__main__":
    # main()
    local_training_no_topology()
