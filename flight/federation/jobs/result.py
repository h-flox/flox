import typing as t

from proxystore.proxy import Proxy
from pydantic.dataclasses import dataclass

from flight.learning.module import RecordList
from flight.strategies import Params

NodeID: t.TypeAlias = t.Hashable
NodeState: t.TypeAlias = tuple


# TODO: Remove config when all type definitions have been resolvedß
@dataclass(config={"arbitrary_types_allowed": True})
class Result:
    state: NodeState
    node_idx: NodeID
    params: Params
    records: RecordList
    cache: dict[str, t.Any]


AbstractResult: t.TypeAlias = Result | Proxy[Result]
"""Helper type alias for a `Result` or a proxy to a `Result`."""
