import typing as t
from concurrent.futures import Future
from dataclasses import dataclass

import pydantic as pyd

from flight.federation.topologies.node import Node
from flight.learning.module import RecordList

if t.TYPE_CHECKING:
    from flight.learning.datasets import DataLoadable
    from flight.learning.module import FlightModule
    from flight.strategies.trainer import TrainerStrategy
    from flight.strategies.worker import WorkerStrategy

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
    ...
    """

    node: Node
    parent: Node
    model: FlightModule
    data: DataLoadable
    worker_strategy: WorkerStrategy
    trainer_strategy: TrainerStrategy


AggrJob: t.TypeAlias = t.Callable[[AggrJobArgs], Result]
TrainJob: t.TypeAlias = t.Callable[[TrainJobArgs], Result]
