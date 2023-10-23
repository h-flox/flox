from __future__ import annotations

from torch.utils.data import Dataset, Subset
from typing import Mapping

from flox.flock import FlockNodeID
from flox.utils.data import FederatedDataset
from flox.utils.data.base import T_co


class FederatedSubsets(FederatedDataset):
    def __init__(self, mapping: Mapping[FlockNodeID, Dataset[T_co] | Subset[T_co]]):
        super().__init__()
        self.mapping = mapping

    def load(self, idx: FlockNodeID) -> Dataset[T_co] | Subset[T_co]:
        """
        Get the subset of data associated with the Flock node identified by ``idx``.

        Args:
            idx (FlockNodeID): The identifier of the Flock node.

        Returns:
            The subset of data the identified Flock node owns.
        """
        return self.mapping[idx]

    def __len__(self) -> int:
        """
        Gets the number of worker nodes with a subset of the data (i.e., the length of
        the ``mapping`` data member).

        Returns:
            Number of worker nodes with subsets according to ``self.mapping``.
        """
        return len(self.mapping)
