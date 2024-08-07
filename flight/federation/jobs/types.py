from __future__ import annotations

import typing as t
from concurrent.futures import Future
from dataclasses import dataclass, field

from proxystore.proxy import Proxy

from flight.federation.topologies.node import Node, NodeState, WorkerState
from flight.learning.modules.prototypes import Record
from flight.learning.modules.torch import FlightModule, TorchDataModule
from flight.learning.types import Params

if t.TYPE_CHECKING:
    from flight.strategies.trainer import TrainerStrategy
    from flight.strategies.worker import WorkerStrategy


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


AbstractResult: t.TypeAlias = Result | Proxy[Result]
"""
Helper type alias for a `Result` or a proxy to a `Result`.
"""


# class TrainJob(t.Protocol):
#     @staticmethod
#     def __call__(
#         node: Node,
#         parent: Node,
#         model: FlightModule,
#         data: DataLoadable,
#         worker_strategy: WorkerStrategy,
#         trainer_strategy: TrainerStrategy,
#     ) -> Result:
#         pass


@dataclass(slots=True, frozen=True)
class AggrJobArgs:
    future: Future
    children: t.Any
    children_futures: t.Sequence[Future]


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
    model: FlightModule | None  # TODO: May need to remove the `None` type.
    data: TorchDataModule  # DataLoadable
    worker_strategy: WorkerStrategy
    trainer_strategy: TrainerStrategy


AggrJob: t.TypeAlias = t.Callable[[AggrJobArgs], Result]
"""
Function signature for aggregation jobs.
"""

TrainJob: t.TypeAlias = t.Callable[[TrainJobArgs], Result]
"""
Function signature for loca training jobs.
"""
