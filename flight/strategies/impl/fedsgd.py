from __future__ import annotations

import typing as t

from numpy.random import Generator

from flight.strategies import Strategy
from flight.strategies.base import (
    DefaultAggrStrategy,
    DefaultCoordStrategy,
    DefaultTrainerStrategy,
    DefaultWorkerStrategy,
)
from flight.strategies.commons import average_state_dicts, random_worker_selection

if t.TYPE_CHECKING:
    from flight.federation.topologies.node import Node, NodeID
    from flight.strategies import NodeState, Params


class FedSGDCoord(DefaultCoordStrategy):
    def __init__(
        self,
        participation,
        probabilistic,
        always_include_child_aggregators: bool,
    ):
        self.participation = participation
        self.probabilistic = probabilistic
        self.always_include_child_aggregators = always_include_child_aggregators

    def select_worker_nodes(
        self, state: NodeState, workers: t.Iterable[Node], rng: Generator | None = None
    ) -> t.Sequence[Node]:
        selected_workers = random_worker_selection(
            workers,
            participation=self.participation,
            probabilistic=self.probabilistic,
            always_include_child_aggregators=self.always_include_child_aggregators,
            rng=rng,
        )
        return selected_workers


class FedSGDAggr(DefaultAggrStrategy):
    def aggregate_params(
        self,
        state: NodeState,
        children_states: t.Mapping[NodeID, NodeState],
        children_state_dicts: t.Mapping[NodeID, Params],
        **kwargs,
    ) -> Params:
        return average_state_dicts(children_state_dicts, weights=None)


class FedSGD(Strategy):
    def __init__(
        self,
        participation: float = 1.0,
        probabilistic: bool = False,
        always_include_child_aggregators: bool = True,
    ):
        super().__init__(
            coord_strategy=FedSGDCoord(
                participation,
                probabilistic,
                always_include_child_aggregators,
            ),
            aggr_strategy=FedSGDAggr(),
            worker_strategy=DefaultWorkerStrategy(),
            trainer_strategy=DefaultTrainerStrategy(),
        )
