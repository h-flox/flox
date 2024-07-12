from __future__ import annotations

import typing as t

from flight.strategies.base import (
    DefaultAggrStrategy,
    DefaultTrainerStrategy,
    DefaultWorkerStrategy,
    Strategy,
)
from flight.strategies.commons import average_state_dicts

from .fedsgd import FedSGDCoord

if t.TYPE_CHECKING:
    NodeState: t.TypeAlias = t.Any
    from flight.federation.topologies.node import NodeID
    from flight.strategies import Params


class FedAvgAggr(DefaultAggrStrategy):
    def aggregate_params(
        self,
        state: NodeState,
        children_states: t.Mapping[NodeID, NodeState],
        children_state_dicts: t.Mapping[NodeID, Params],
        **kwargs,
    ) -> Params:
        weights = {}
        for node, child_state in children_states.items():
            weights[node] = child_state["num_data_samples"]

        state["num_data_samples"] = sum(weights.values())

        return average_state_dicts(children_state_dicts, weights=weights)


class FedAvgWorker(DefaultWorkerStrategy):
    def before_training(
        self, state: NodeState, data: Params
    ) -> tuple[NodeState, Params]:
        state["num_data_samples"] = len(data)
        return state, data


class FedAvg(Strategy):
    def __init__(
        self,
        participation: float = 1.0,
        probabilistic: bool = False,
        always_include_child_aggregators: bool = False,
    ):
        super().__init__(
            coord_strategy=FedSGDCoord(
                participation, probabilistic, always_include_child_aggregators
            ),
            aggr_strategy=FedAvgAggr(),
            worker_strategy=FedAvgWorker(),
            trainer_strategy=DefaultTrainerStrategy(),
        )
