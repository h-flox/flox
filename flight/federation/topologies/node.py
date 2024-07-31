import typing as t
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID

import pydantic as pyd

from flight.learning.modules.base import Trainable

NodeID: t.TypeAlias = t.Union[int, str]
"""
ID of nodes in Flight topologies; can either be of type `int` or `str`.
"""


class NodeKind(str, Enum):
    """
    Kinds of nodes that can exist in a Flight topology:
    Coordinator (COORD), Aggregator (AGGR), and Worker (WORKER) nodes.
    """

    COORD = "coordinator"
    AGGR = "aggregator"
    WORKER = "worker"


class Node(pyd.BaseModel):
    """A `Node` in Flight.

    An individual `Node` characterizes an endpoint that either takes on the task of aggregating
    model parameters or performing local training. Their connections are established with the
    [`Topology`][flight.federation.topologies.topo.Topology] class."""

    idx: NodeID
    """
    The ID of the node.
    """

    kind: NodeKind
    """
    The kind of Node---indicates its *role* in a federation.
    """

    globus_comp_id: UUID | None = pyd.Field(default=None)
    """
    Globus Compute UUID for remote execution.
    """

    proxystore_id: UUID | None = pyd.Field(default=None)
    """
    ProxyStore UUID for data transfer for remote execution with Globus Compute.
    """

    extra: dict[str, t.Any] = pyd.Field(default_factory=dict)
    """
    Any extra parameters users wish to give to Nodes (e.g., parameters or settings around
    system resource use).
    """


@dataclass
class NodeState:
    """
    Dataclass that wraps the state of a node during a federation.

    Args:
        idx (NodeID): The ID of the node.

    Throws:
        - TypeError: This class cannot be directly instantiated. Only its children classes can be instantiated.
    """

    idx: NodeID
    cache: dict[str, t.Any] = field(
        init=False, default_factory=dict, repr=False, hash=False
    )

    def __post_init__(self):
        if type(self) is NodeState:
            raise TypeError(
                "Cannot instantiate an instance of `NodeState`. "
                "Instead, you must instantiate instances of `WorkerState` or `AggrState`."
            )


@dataclass
class AggrState(NodeState):
    """
    The state of an Aggregator node.

    Args:
        children (t.Iterable[Node]): Child nodes in the topology.
        aggr_model (t.Optional[Trainable]): Aggregated model.
    """

    children: t.Iterable[Node]
    aggr_model: t.Optional[Trainable] = None


@dataclass
class WorkerState(NodeState):
    """
    The state of a Worker node.

    Args:
        global_model (t.Optional[Trainable]): ...
        local_model (t.Optional[Trainable]): ...
    """

    global_model: t.Optional[Trainable] = None
    local_model: t.Optional[Trainable] = None
