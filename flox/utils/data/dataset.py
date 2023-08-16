from flox.flock import FlockNodeID
from typing import Mapping, TypeVar

D = TypeVar("D")


class FederatedDataset:
    def __init__(self, mapping: Mapping[FlockNodeID, D]):
        if mapping is None:
            mapping = {}
        self._mapping: Mapping[FlockNodeID, D] = mapping

    def __getitem__(self, node: FlockNodeID):
        return self._mapping[node]

    def __setitem__(self, node: FlockNodeID, data: D):
        self._mapping[node] = data
