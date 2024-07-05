import typing as t
from enum import Enum
from uuid import UUID

import pydantic as pyd

NodeID: t.TypeAlias = int | str
"""ID of nodes in Flight topologies; can either be of type `int` or `str`."""


class NodeKind(str, Enum):
    """
    Kinds of nodes that can exist in a Flight topology:
    Coordinator (COORD), Aggregator (AGGR), and Worker (WORKER) nodes.
    """

    COORD = "Coordinator"
    AGGR = "Aggregator"
    WORKER = "Worker"


class Node(pyd.BaseModel):
    idx: NodeID
    kind: NodeKind
    globus_comp_id: UUID | None = pyd.Field(default=None)
    proxystore_id: UUID | None = pyd.Field(default=None)
    extra: dict[str, t.Any] | None = pyd.Field(default=None)
