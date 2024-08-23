from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from proxystore.proxy import Proxy

from flight.federation.topologies.node import Node, NodeState, WorkerState
from flight.learning.base import AbstractDataModule, AbstractModule
from flight.learning.modules.prototypes import Record
from flight.learning.types import Params

if t.TYPE_CHECKING:
    from flight.engine.data import TransferProto
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
    children: t.Iterable[Node]
    child_results: t.Iterable[Result]
    aggr_strategy: AggrStrategy
    transfer: TransferProto


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
    model: AbstractModule | None  # TorchModule | None
    data: AbstractDataModule  # TorchDataModule  # DataLoadable
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
