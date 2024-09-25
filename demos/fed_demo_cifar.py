from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from flight.federation.fed_sync import SyncFederation
from flight.learning.torch import TorchDataModule, TorchModule
from flight.strategies.impl.fedavg import FedAvg
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from flight.federation import Topology
from flight.federation.topologies import Node


class MyCifarModule(TorchModule):
    """Neural network for Flight to learn on Cifar-10."""

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

    def training_step(self, batch, batch_nb):
        inputs, targets = batch
        preds = self(inputs)
        loss = self.criterion(preds, targets)
        return loss

    def validation_step(self, batch, batch_nb):
        return self.training_step(batch, batch_nb)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.001)


class CifarDataModule(TorchDataModule):
    """Fligth data module for loading the Cifar-10 data."""

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
        subset_range = self._get_subset(node.idx)
        subset = Subset(
            self._train_data,
            indices=list(subset_range),
        )
        return DataLoader(subset, self.batch_size)

    def size(self, state) -> int:  # node: Node | None = None) -> int:
        # subset_range = self._get_subset(state.idx)
        subset_range = self._get_subset(state)
        return len(subset_range)

    def _get_subset(self, node_idx):
        match node_idx:
            case 1:
                start, end = 0, 5_000  # 1000
            case 2:
                start, end = 5_000, 10_000
            case 3:
                start, end = 10_000, 15_000
            case _:
                raise ValueError(f"Illegal value {node_idx=}")

        return range(start, end)

    # def valid_data(self, node: Node | None = None) -> DataLoader:
    #     subset = Subset(
    #         self._train_data,
    #         indices=list(range(45_000, len(self._train_data))),
    #     )
    #     return DataLoader(subset, self.batch_size)
    #
    # def test_data(self, node: Node | None = None) -> DataLoader:
    #     return DataLoader(self._test_data, self.batch_size)


def main():
    topology = Topology.from_dict(
        {
            0: {"kind": "coordinator", "children": [1, 2, 3]},
            1: {"kind": "worker", "children": [], "extra": {"device": "mps"}},
            2: {"kind": "worker", "children": [], "extra": {"device": "mps"}},
            3: {"kind": "worker", "children": [], "extra": {"device": "mps"}},
        }
    )

    strategy = FedAvg()
    module = MyCifarModule()
    data = CifarDataModule("~/Research/Data/Torch-Data/")
    federation = SyncFederation(topology, strategy, module, data)
    results = federation.start(40)
    print("Finished!!")

    records = []
    for res in results:
        records.extend(res.records)

    df = pd.DataFrame.from_records(records)
    df.to_feather("tmp-cifar.feather")

    sns.lineplot(df, x="round", y="train/loss")
    plt.show()


if __name__ == "__main__":
    main()
