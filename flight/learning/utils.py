from __future__ import annotations

import typing as t

from .base import AbstractDataModule, Data

if t.TYPE_CHECKING:
    from ..federation.topologies.node import Node, NodeID


class FederatedDataModule(AbstractDataModule):
    def __init__(
        self,
        dataset: Data,
        train_indices: t.Mapping[NodeID, t.Iterable[int]],
        test_indices: t.Mapping[NodeID, t.Iterable[int]] | None,
        valid_indices: t.Mapping[NodeID, t.Iterable[int]] | None,
    ) -> None:
        self.dataset = dataset
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.valid_indices = valid_indices

    def train_data(self, node: Node | None = None) -> Data:
        # TODO
        pass
