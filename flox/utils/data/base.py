from __future__ import annotations

from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import TypeVar

from flox.flock import FlockNodeID

T_co = TypeVar("T_co", covariant=True)


class FederatedDataset(ABC):
    """
    Base class for all types of federated datasets.
    """

    def __init__(self):
        pass

    def __getitem__(self, idx: FlockNodeID) -> Dataset[T_co]:
        """Convenience function that does the same as ``flox.utils.data.FederatedDataset.load()``. This should not
           be overridden by child classes.

        Args:
            idx (FlockNodeID):

        Returns:

        """
        return self.load(idx)

    @abstractmethod
    def load(self, idx: FlockNodeID) -> Dataset[T_co]:
        """
        ...

        Args:
            idx (FlockNodeID): ...

        Returns:
            ...
        """
