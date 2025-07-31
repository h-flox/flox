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

from v1.flight import Topology
from v1.flight.topologies import Node
from v1.flight.fit import Topology, federated_fit
from v1.flight.learning.torch import TorchDataModule, TorchModule
from torchvision.transforms import ToTensor


class MyMnistModule(TorchModule):
    """Neural network for Flight to learn on MNIST."""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
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
        
        self._train_data = MNIST(
            root=".", train=True, download=False, transform=ToTensor()
        )
        self._test_data = MNIST(
            root=".", train=False, download=False, transform=ToTensor()
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
