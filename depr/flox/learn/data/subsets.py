from __future__ import annotations

import typing as t

from torch.utils.data import Dataset, Subset

from flox.federation.topologies import Node, NodeID
from flox.learn.data import FloxDataset

if t.TYPE_CHECKING:
    from flox.learn.data.types import T_co


class FederatedSubsets(FloxDataset):
    """
    A subset...
    """

    def __init__(self, dataset: Dataset[T_co], indices: dict[Node, list[int]]):
        super().__init__()
        self.dataset = dataset
        self.node_indices = indices
        self._num_subsets = len(list(self.node_indices))

    def __getitem__(self, node: Node) -> Subset[T_co]:
        if isinstance(node, NodeID):
            raise ValueError("Indexing is done by the Node object itself.")

        return self.load(node)

    def __len__(self):
        return self._num_subsets

    def __iter__(self) -> t.Iterator[tuple[NodeID, Subset[T_co]]]:
        for node in self.node_indices:
            yield node.idx, self.load(node)

    def load(self, node: Node) -> Subset[T_co]:
        return Subset(self.dataset, self.node_indices[node.idx])

    @property
    def number_of_subsets(self):
        return self._num_subsets
