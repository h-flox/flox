import typing as t

from ...federation.topologies.node import Node


class DataLoadable(t.Protocol):
    def load(self, node: Node):
        pass
