from __future__ import annotations

import abc
import typing as t

from torch.utils.data import Dataset, Subset

from flox.topos import Node

if t.TYPE_CHECKING:
    from flox.topos import NodeID
    from flox.topos.states import NodeState

    T_co = t.TypeVar("T_co", covariant=True)


class FloxDataset(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def load(self, node: Node | NodeID):
        pass


class FederatedSubsets(FloxDataset):
    """
    A subset...
    """

    def __init__(self, dataset: Dataset[T_co], indices: dict[NodeID, list[int]]):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self._num_subsets = len(list(self.indices))

    def load(self, node: Node | NodeID) -> Subset[T_co]:
        if isinstance(node, Node):
            node = node.idx
        return Subset(self.dataset, self.indices[node])

    @property
    def number_of_subsets(self):
        return self._num_subsets

    def __getitem__(self, node: Node | NodeID) -> Subset[T_co]:
        return self.load(node)

    def __len__(self):
        return self._num_subsets

    def __iter__(self) -> t.Iterator[tuple[NodeID, Subset[T_co]]]:
        for idx in self.indices:
            yield idx, self.load(idx)


class LocalDataset(FloxDataset):
    """
    Local dataset...
    """

    def __init__(self, state: NodeState, /, *args, **kwargs):
        super().__init__()

    def load(self, node: Node | NodeID) -> Dataset[T_co]:
        """Loads local dataset into a PyTorch object.

        Args:
            node (Node | NodeID): ...

        Returns:
            Dataset object using local data.
        """
        raise NotImplementedError
