from __future__ import annotations

import typing
from dataclasses import dataclass

from proxystore.proxy import Proxy

if typing.TYPE_CHECKING:
    from pandas import DataFrame

    from flox.flock import FlockNodeID, FlockNodeKind
    from flox.flock.states import NodeState
    from flox.nn.typing import StateDict


@dataclass
class JobResult:
    """A simple dataclass that is returned by jobs executed on Aggregator and Worker nodes in a ``Flock``.

    Aggregators and Worker nodes have to return the same type of object to support hierarchical execution.
    """

    node_state: NodeState | None
    """The state of the ``Flock`` node based on its kind."""

    node_idx: FlockNodeID | None
    """The ID of the ``Flock`` node."""

    node_kind: FlockNodeKind | None
    """The kind of the ``Flock`` node."""

    state_dict: StateDict | None
    """The ``StateDict`` of the PyTorch global_module (either aggregated or trained locally)."""

    history: DataFrame | None
    """The history of results."""

    cache: dict[str, typing.Any] | None = None
    """Miscellaneous data to be returned as part of the ``JobResult``."""


Result: typing.TypeAlias = JobResult | Proxy[JobResult]
