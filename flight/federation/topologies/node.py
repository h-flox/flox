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

    COORD = "coordinator"
    AGGR = "aggregator"
    WORKER = "worker"


class Node(pyd.BaseModel):
    """A `Node` in Flight.

    An individual `Node` characterizes an endpoint that either takes on the task of aggregating
    model parameters or performing local training. Their connections are established with the
    [`Topology`][flight.federation.topologies.topo.Topology] class."""

    idx: NodeID
    """The ID of the node."""
    kind: NodeKind
    """The kind of Node---indicates its *role* in a federation."""
    globus_comp_id: UUID | None = pyd.Field(default=None)
    """Globus Compute UUID for remote execution."""
    proxystore_id: UUID | None = pyd.Field(default=None)
    """ProxyStore UUID for data transfer for remote execution with Globus Compute."""
    extra: dict[str, t.Any] | None = pyd.Field(default=None)
    """Any extra parameters users wish to give to Nodes (e.g., parameters or settings around system resource use)."""
