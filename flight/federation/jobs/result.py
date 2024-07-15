import typing as t

from proxystore.proxy import Proxy
from pydantic.dataclasses import dataclass

from flight.learning.module import RecordList
from flight.strategies.aggr import Params

if t.TYPE_CHECKING:
    NodeID: t.TypeAlias = t.Hashable
    NodeState: t.TypeAlias = tuple


@dataclass
class Result:
    state: NodeState
    node_idx: NodeID
    params: Params
    records: RecordList
    cache: dict[str, t.Any]


AbstractResult: t.TypeAlias = Result | Proxy[Result]
"""Helper type alias for a `Result` or a proxy to a `Result`."""
