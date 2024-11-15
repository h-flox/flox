from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from proxystore.proxy import Proxy

from flight.federation.topologies.node import Node, NodeState, WorkerState
from flight.learning.base import AbstractDataModule, AbstractModule

if t.TYPE_CHECKING:
    from flight.types import Record
    from flight.engine.transporters import AbstractTransporter
    from flight.learning.params import Params
    from flight.strategies import AggrStrategy, TrainerStrategy, WorkerStrategy


@dataclass
class Result:
    node: Node
    """
    The node that produced this result during a federation.
    """

    node_state: NodeState
    """
    The current state of the node that returned a given result during a federation.
    """

    params: Params
    """
    Parameters returned as part of a result from a single Node in a federation.
    """

    module: AbstractModule | None = field(default=None)
    """
    The model module that was returned as part of this result. For *Workers*, these
    are locally-trained model updates. For *Aggregators*, these are the aggregated
    modules.
    """

    records: list[Record] = field(default_factory=list)
    """
    List of records for model training/aggregation metrics.
    """

    extra: dict[str, t.Any] = field(default_factory=dict)
    """
    Extra data recorded by a node during the runtime of its job.
    """


@dataclass(slots=True, frozen=True)
class AggrJobArgs:
    # fut: Future
    round_num: int
    node: Node
    children: t.Sequence[Node]
    child_results: t.Sequence[Result]
    aggr_strategy: AggrStrategy
    transfer: AbstractTransporter


@dataclass(slots=True, frozen=True)
class TrainJobArgs:
    """
    Arguments for the local training job run on worker nodes in a federation.

    The default implementation for a local training job is given by
    [`default_training_job`][flight.federation.jobs.work.default_training_job].
    """

    node: Node
    parent: Node
    node_state: WorkerState
    model: AbstractModule | None
    data: AbstractDataModule
    worker_strategy: WorkerStrategy
    trainer_strategy: TrainerStrategy


AbstractResult: t.TypeAlias = Result | Proxy[Result]
"""
Helper type alias for a `Result` or a proxy to a `Result`.
"""

AggrJob: t.TypeAlias = t.Callable[[AggrJobArgs], Result]
"""
Function signature for aggregation jobs.
"""

TrainJob: t.TypeAlias = t.Callable[[TrainJobArgs], Result]
"""
Function signature for loca training jobs.
"""
