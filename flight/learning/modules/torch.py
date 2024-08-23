from __future__ import annotations

import abc
import typing as t

from torch.utils.data import DataLoader

if t.TYPE_CHECKING:
    from flight.federation.topologies import Node


class TorchDataModule(abc.ABC):
    @abc.abstractmethod
    def train_data(self, node: Node | None = None) -> DataLoader:
        pass

    # noinspection PyMethodMayBeStatic
    def test_data(self, node: Node | None = None) -> DataLoader | None:
        return None

    def valid_data(self, node: Node | None = None) -> DataLoader | None:
        return None

    def size(self, node: Node | None = None) -> int | None:
        return None
