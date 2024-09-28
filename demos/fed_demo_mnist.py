import os
import sys

sys.path.append(os.getcwd())

import os
import sys

sys.path.append(os.getcwd())

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

from flight.federation import Topology
from flight.federation.topologies import Node
from flight.fit import federated_fit
from flight.learning.torch import TorchDataModule, TorchModule


class MyMnistModule(TorchModule):
    """Neural network for Flight to learn on MNIST."""

    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 28 * 28 * 28),
            nn.ReLU(),
            nn.Linear(28 * 28 * 28, 28 * 28),
            nn.ReLU(),
            nn.Linear(28 * 28, 10),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x
        return self.model(x)

    def training_step(self, batch, batch_nb):
        inputs, targets = batch
        preds = self(inputs)
        loss = self.criterion(preds, targets)
        return loss

    def validation_step(self, batch, batch_nb):
        return self.training_step(batch, batch_nb)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.001)


class MnistDataModule(TorchDataModule):
    """Flight data module for loading the MNIST data."""

    def __init__(
        self,
        nodes: list[Node],
        root: Path | str,
        data_size: int | None = None,
        batch_size: int = 128,
    ):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        )
        self._train_data = MNIST(
            root=root, train=True, download=False, transform=transform
        )
        self._test_data = MNIST(
            root=root, train=False, download=False, transform=transform
        )

        if data_size is not None:
            data_size = min(data_size, len(self._train_data))
            self._train_data = Subset(self._train_data, list(range(data_size)))

        self.node_ids = [node.idx for node in nodes]
        self.num_nodes = len(self.node_ids)
        self.num_subsets = {}

        n = len(self._train_data)
        data_per_node = n // self.num_nodes
        for i, idx in enumerate(self.node_ids):
            self.num_subsets[idx] = (i * data_per_node, (i + 1) * data_per_node)

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
        start, end = self.num_subsets[node_idx]
        return range(start, end)


def main():
    module = MyMnistModule()
    topo = Topology.from_yaml("demos/topo.yaml")

    data = MnistDataModule(
        list(topo.nodes()),
        "~/Research/Data/Torch-Data/",
        data_size=1_000,
    )
    trained_module, results = federated_fit(
        topo,
        module,
        data,
        rounds=10,
    )

    records = []
    for res in results:
        records.extend(res.records)

    df = pd.DataFrame.from_records(records)
    sns.lineplot(df, x="round", y="train/loss")
    plt.show()
    df.to_feather("tmp.feather")


if __name__ == "__main__":
    main()
