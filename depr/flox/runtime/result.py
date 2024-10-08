from __future__ import annotations

import random
import typing as t
from concurrent.futures import Future
from dataclasses import dataclass, field

from proxystore.proxy import Proxy

if t.TYPE_CHECKING:
    from pandas import DataFrame

    from flox.federation.topologies import NodeID, NodeKind, NodeState
    from flox.learn.types import Params


@dataclass
class JobResult:
    """A simple dataclass that is returned by jobs executed on Aggregator and Worker nodes in a ``Flock``.

    Aggregators and Worker nodes have to return the same type of object to support hierarchical execution.
    """

    node_state: NodeState
    """The state of the ``Flock`` node based on its kind."""

    node_idx: NodeID
    """The ID of the ``Flock`` node."""

    node_kind: NodeKind
    """The kind of the ``Flock`` node."""

    params: Params
    """The ``Params`` of the PyTorch global_model (either aggregated or trained locally)."""

    history: DataFrame
    """The history of results."""

    cache: dict[str, t.Any] = field(init=False, default_factory=dict)
    """Miscellaneous data to be returned as part of the ``JobResult``."""

    def __hash__(self):
        """TODO: Resolve this hacky fix related to ProxyStore."""
        return random.randint(0, 1000000)


Result: t.TypeAlias = JobResult | Proxy[JobResult]
"""The result of a job or the proxied result of a job (if using
[ProxyStoreTransfer][flox.runtime.transfer.proxystore.ProxyStoreTransfer])."""

ResultFuture: t.TypeAlias = Future[Result]
"""A Future object that will return a Result when finished."""
