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
    """The coordinator and its respective methods for 'FedSGD'."""

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
        """Method containing the method for worker selection for 'FedSGD'.

        Args:
            state (NodeState): State of the coordinator node.
            workers (t.Iterable[Node]): Iterable containing the worker nodes.
            rng (Generator | None, optional): RNG object used for randomness. Defaults to None.

        Returns:
            t.Sequence[Node]: The selected worker nodes.
        """
        selected_workers = random_worker_selection(
            workers,
            participation=self.participation,
            probabilistic=self.probabilistic,
            always_include_child_aggregators=self.always_include_child_aggregators,
            rng=rng,
        )
        return selected_workers


class FedSGDAggr(DefaultAggrStrategy):
    """The aggregator and its respective methods for 'FedSGD'.

    Args:
        DefaultAggrStrategy: The base class providing the necessary methods for 'FedSGDAggr'.
    """

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
        return average_state_dicts(children_state_dicts, weights=None)


class FedSGD(Strategy):
    """
    Implementation of the FedSGD strategy, which uses 'FedSGD' for the coordinator and aggregators, and defaults
    for the workers and trainer.
    """

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
