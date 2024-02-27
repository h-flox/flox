from __future__ import annotations

import abc
import typing

from torch.utils.data import Dataset, Subset

from flox.flock import FlockNode

if typing.TYPE_CHECKING:
    from typing import TypeVar, Iterator

    from flox.flock.states import NodeState
    from flox.flock import FlockNodeID

    T_co = TypeVar("T_co", covariant=True)


class FloxDataset(abc.ABC):
    """
    Abstract...
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def load(self, node: FlockNode | FlockNodeID):
        pass


class FederatedSubsets(FloxDataset):
    """
    A subset...
    """

    def __init__(self, dataset: Dataset[T_co], indices: dict[FlockNodeID, list[int]]):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self._num_subsets = len(list(self.indices))

    def load(self, node: FlockNode | FlockNodeID) -> Subset[T_co]:
        if isinstance(node, FlockNode):
            node = node.idx
        return Subset(self.dataset, self.indices[node])

    @property
    def number_of_subsets(self):
        return self._num_subsets

    def __getitem__(self, node: FlockNode | FlockNodeID) -> Subset[T_co]:
        return self.load(node)

    def __len__(self):
        return self._num_subsets

    def __iter__(self) -> Iterator[tuple[FlockNodeID, Subset[T_co]]]:
        for idx in self.indices:
            yield idx, self.load(idx)


class LocalDataset(FloxDataset):
    """
    Local dataset...
    """

    def __init__(self, state: NodeState, /, *args, **kwargs):
        super().__init__()

    def load(self, node: FlockNode | FlockNodeID) -> Dataset[T_co]:
        """Loads local dataset into a PyTorch object.

        Args:
            node (FlockNode | FlockNodeID): ...

        Returns:
            Dataset object using local data.
        """
        raise NotImplementedError
