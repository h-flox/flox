from __future__ import annotations

import typing as t

from torch.utils.data import Dataset

from flox.federation.topologies import Node, NodeState

from .core import FloxDataset

if t.TYPE_CHECKING:
    from .types import T_co


class LocalDataset(FloxDataset):
    """
    Local dataset...
    """

    def __init__(self, state: NodeState, /, *args, **kwargs):
        super().__init__()

    def load(self, node: Node) -> Dataset[T_co]:
        """Loads local dataset into a PyTorch object.

        Args:
            node (Node | NodeID): ...

        Returns:
            Dataset object using local data.
        """
        raise NotImplementedError
