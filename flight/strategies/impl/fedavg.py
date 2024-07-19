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
    from flight.federation.topologies.node import NodeID
    from flight.strategies import NodeState, Params


class FedAvgAggr(DefaultAggrStrategy):
    """The aggregator for the FedAvg algorithm and its respective methods."""

    def aggregate_params(
        self,
        state: NodeState,
        children_states: t.Mapping[NodeID, NodeState],
        children_state_dicts: t.Mapping[NodeID, Params],
        **kwargs,
    ) -> Params:
        """Method used by aggregator nodes for aggregating the passed node state dictionary.

        Args:
            state (NodeState): State of the current aggregator node.
            children_states (t.Mapping[NodeID, NodeState]): Dictionary of the states of the children.
            children_state_dicts (t.Mapping[NodeID, Params]): Dictionary mapping each child to its values.
            **kwargs: Key word arguments provided by the user.

        Returns:
            Params: The aggregated values.
        """
        weights = {}
        for node, child_state in children_states.items():
            weights[node] = child_state["num_data_samples"]

        state["num_data_samples"] = sum(weights.values())

        return average_state_dicts(children_state_dicts, weights=weights)


class FedAvgWorker(DefaultWorkerStrategy):
    """The worker for 'FedAvg' and its respective methods.

    Args:
        DefaultWorkerStrategy: The base class providing necessary methods for 'FedAvgWorker'
    """

    def before_training(
        self, state: NodeState, data: Params
    ) -> tuple[NodeState, Params]:
        """Callback to run before the current nodes training.

        Args:
            state (NodeState): State of the current worker node.
            data (Params): The data related to the current worker node.

        Returns:
            tuple[NodeState, Params]: A tuple containing the updated state of the worker node and the data.
        """
        state["num_data_samples"] = len(data)
        return state, data


class FedAvg(Strategy):
    """Implementation of the FedAvg strategy, which uses default strategies for the trainer,
        'FedAvg' for aggregator and workers, and 'FedSGD' for the coordinator.

    Args:
        Strategy: The base class providing the necessary attributes for 'FedAvg'.
    """

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
