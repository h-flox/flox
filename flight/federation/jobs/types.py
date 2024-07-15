import typing as t

import pydantic as pyd

from flight.federation.topologies.node import Node
from flight.learning.module import RecordList

if t.TYPE_CHECKING:
    NodeState: t.TypeAlias = t.Any
    Params: t.TypeAlias = t.Any


@pyd.dataclasses.dataclass
class Result(pyd.BaseModel):
    node: Node = pyd.Field()
    node_state: NodeState = pyd.Field()
    params: Params = pyd.Field()
    records: RecordList = pyd.Field()
    cache: dict[str, t.Any] = pyd.Field(default_factory=dict, init=False)


AggrJob: t.TypeAlias = t.Callable[[Node, Node], Result]
TrainJob: t.TypeAlias = t.Callable[[Node, Node], Result]
