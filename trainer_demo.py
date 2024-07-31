import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flight.federation.topologies import Node
from flight.federation.topologies.node import WorkerState
from flight.learning.modules.torch import FlightModule, TorchDataModule
from flight.learning.trainers.torch import TorchTrainer
from flight.learning.types import LocalStepOutput
from flight.strategies.base import DefaultTrainerStrategy


class MyModule(FlightModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb) -> LocalStepOutput:
        inputs, targets = batch
        preds = self(inputs)
        loss = F.l1_loss(preds, targets)
        return loss

    def validation_step(self, batch, batch_nb) -> LocalStepOutput:
        return self.training_step(batch, batch_nb)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.001)


class MyDataModule(TorchDataModule):
    def __init__(
        self,
        num_samples: int = 10_000,
        num_features: int = 1,
        seed: int = 0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        x = torch.randn((num_samples, num_features))
        y = torch.randn((num_samples, 1))
        self.raw_data = TensorDataset(x, y)

    def train_data(self, node: Node | None = None) -> DataLoader:
        return DataLoader(self.raw_data, batch_size=32)

    def valid_data(self, node: Node | None = None) -> DataLoader:
        return self.train_data(node)


if __name__ == "__main__":
    state = WorkerState(0, None, None)

    module = MyModule()
    data = MyDataModule()

    trainer = TorchTrainer(
        Node(idx=0, kind="worker"), DefaultTrainerStrategy(), max_epochs=10
    )
    results = trainer.fit(state, module, data)

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame.from_records(results)
    sns.lineplot(df, x="epoch", y="train/loss")
    sns.lineplot(df, x="epoch", y="val/loss")
    plt.show()
