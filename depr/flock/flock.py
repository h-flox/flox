import networkx as nx

from enum import Enum, auto
from uuid import UUID
from typing import Optional, Iterable


class EndpointKind(Enum):
    aggr = auto()
    worker = auto()


class EndpointAttrs(Enum):
    pass


class Flock:
    topo: nx.DiGraph

    def __init__(self):
        pass

    def add_endpoint(
            self,
            kind: EndpointKind,
            indices: Optional[list[int]] = None,
            addr: Optional[UUID] = None
    ):
        pass

    def remove_endpoint(
            self,
            idx
    ):
        pass

    @property
    def leader(self):
        raise NotImplementedError()

    @property
    def aggregators(self) -> Iterable:
        raise NotImplementedError()

    @property
    def workers(self) -> Iterable:
        raise NotImplementedError()

    # ---------------------------------------------------------------------------- #

    @classmethod
    def from_yaml(cls):
        pass

    @classmethod
    def from_json(cls):
        pass

    @classmethod
    def from_gml(cls):
        pass

    @classmethod
    def from_graphml(cls):
        pass

    @classmethod
    def from_gexf(cls):
        pass

    @classmethod
    def from_pajek(cls):
        pass

    @classmethod
    def from_networkx(cls):
        pass
