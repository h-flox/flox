from __future__ import annotations

import typing as t

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from flight.federation.topologies import Node
from flight.learning.modules.torch import TorchDataModule

NODE_DATA_PATH = "node_data_path"

if t.TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def node() -> Node:
    return Node(idx=0, kind="worker")


@pytest.fixture
def data() -> TensorDataset:
    x = torch.tensor([[val] for val in range(100)])
    y = torch.tensor([[xi**2] for xi in x])
    return TensorDataset(x, y)


class InMemoryDataModule(TorchDataModule):
    def __init__(self, data: TensorDataset):
        super().__init__()
        self.data = data

    def train_data(self, node: Node | None = None) -> DataLoader:
        return DataLoader(self.data)


class DiscDataModule(TorchDataModule):
    def __init__(self, root: Path | str | None):
        super().__init__()
        self.root = root

    def train_data(self, node: Node | None = None) -> DataLoader:
        if self.root is None:
            root = node[NODE_DATA_PATH]
        else:
            root = self.root

        with open(root, "rb") as fp:
            data = torch.load(fp)
        return DataLoader(data)


class TestInMemoryData:
    """
    Test suite that tests a ``DataModule`` implementation that loads data from memory.
    """

    def test_read_from_memory(self, data):
        dataset = InMemoryDataModule(data)
        assert isinstance(dataset, TorchDataModule)
        assert isinstance(dataset.train_data(), DataLoader)


class TestDiscData:
    """
    Test suite that tests a ``DataModule`` implementation that loads data from disc.
    """

    def test_read_from_disc(self, data, tmp_path):
        filename = tmp_path / "temp.pt"
        with open(filename, "wb") as fp:
            torch.save(data, fp)

        dataset = DiscDataModule(filename)
        assert isinstance(dataset, TorchDataModule)
        assert isinstance(dataset.train_data(), DataLoader)

    def test_read_from_disc_with_node(self, node, data, tmp_path):
        node[NODE_DATA_PATH] = tmp_path / "temp.pt"
        with open(node[NODE_DATA_PATH], "wb") as fp:
            torch.save(data, fp)

        dataset = DiscDataModule(root=None)
        assert isinstance(dataset, TorchDataModule)
        assert isinstance(dataset.train_data(node), DataLoader)
