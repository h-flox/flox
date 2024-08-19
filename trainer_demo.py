from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from flight.federation.topologies import Node
from flight.federation.topologies.node import WorkerState
from flight.learning.modules.torch import TorchDataModule, TorchModule
from flight.learning.trainers.torch import TorchTrainer
from flight.learning.types import LocalStepOutput
from flight.strategies.base import DefaultTrainerStrategy


class MyModule(TorchModule):
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


class MyCifarModule(TorchModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_nb) -> LocalStepOutput:
        inputs, targets = batch
        preds = self(inputs)
        loss = self.criterion(preds, targets)
        return loss

    def validation_step(self, batch, batch_nb) -> LocalStepOutput:
        return self.training_step(batch, batch_nb)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.001)


class CifarDataModule(TorchDataModule):
    def __init__(self, root: Path | str, batch_size: int = 128):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self._train_data = CIFAR10(
            root=root, train=True, download=False, transform=transform
        )
        self._test_data = CIFAR10(
            root=root, train=False, download=False, transform=transform
        )
        self.batch_size = batch_size

    def train_data(self, node: Node | None = None) -> DataLoader:
        subset = Subset(
            self._train_data,
            indices=list(range(0, 45_000)),
        )
        return DataLoader(subset, self.batch_size)

    def valid_data(self, node: Node | None = None) -> DataLoader:
        subset = Subset(
            self._train_data,
            indices=list(range(45_000, len(self._train_data))),
        )
        return DataLoader(subset, self.batch_size)

    def test_data(self, node: Node | None = None) -> DataLoader:
        return DataLoader(self._test_data, self.batch_size)

    @property
    def classes(self) -> tuple[str, ...]:
        return (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar", action="store_true")
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
    )
    args = parser.parse_args()

    if args.cifar:
        data = CifarDataModule("~/Research/Data/Torch-Data/")
        module = MyCifarModule()
    else:
        data = MyDataModule()
        module = MyModule()

    trainer = TorchTrainer(
        Node(idx=0, kind="worker"),
        DefaultTrainerStrategy(),
        max_epochs=20,
    )
    results = trainer.fit(
        WorkerState(0, None, None),
        module,
        data,
    )

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame.from_records(results)
    sns.lineplot(df, x="epoch", y="train/loss")
    sns.lineplot(df, x="epoch", y="val/loss")
    plt.show()
