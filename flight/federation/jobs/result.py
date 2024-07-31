import typing as t

from proxystore.proxy import Proxy
from pydantic.dataclasses import dataclass

from flight.federation.topologies.node import NodeID, NodeState
from flight.learning.types import Params
from flight.types import Record


# TODO: Remove config when all type definitions have been resolved
@dataclass(config={"arbitrary_types_allowed": True})
class Result:
    state: NodeState
    node_idx: NodeID
    params: Params
    records: list[Record]
    cache: dict[str, t.Any]


AbstractResult: t.TypeAlias = Result | Proxy[Result]
"""
Helper type alias for a `Result` or a proxy to a `Result`.
"""
