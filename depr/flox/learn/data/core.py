from __future__ import annotations

import abc
import typing as t

from flox.federation.topologies import Node

if t.TYPE_CHECKING:
    from torch.utils.data import Dataset

    from .types import T_co


class FloxDataset(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def load(self, node: Node) -> Dataset[T_co]:
        pass
