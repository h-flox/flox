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
    from flight.federation.topologies.node import AggrState, Node, NodeID, NodeState
    from flight.learning.types import Params


class FedSGDCoord(DefaultCoordStrategy):
    """
    The coordinator and its respective methods for 'FedSGD'.
    """

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
        self,
        state: NodeState,
        workers: t.Iterable[Node],
        rng: Generator,
    ) -> t.Sequence[Node]:
        """Method containing the method for worker selection for 'FedSGD'.

        Args:
            state (NodeState): State of the coordinator node.
            workers (t.Iterable[Node]): Iterable containing the worker nodes.
            rng (Generator): Generator used for random sampling (if needed).

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
    """
    Standard averaging strategy.
    """

    def aggregate_params(
        self,
        state: AggrState,
        children_states: t.Mapping[NodeID, NodeState],
        children_state_dicts: t.Mapping[NodeID, Params],
        **kwargs,
    ) -> Params:
        """
        Performs a simple average of the model parameters returned by the child nodes.

        The average is done by:

        $$
            w^{t} \\triangleq \\frac{1}{K} \\sum_{k=1}^{K} w_{k}^{t}
        $$

        where $w^{t}$ is the aggregated model parameters, $K$ is the number of returned
        model updates, $t$ is the current round, and $w_{k}^{t}$ is the returned model
        updates from child $k$ at round $t$.

        Args:
            state (NodeState): State of the current aggregator node.
            children_states (t.Mapping[NodeID, NodeState]): Dictionary of the states of
                the children.
            children_state_dicts (t.Mapping[NodeID, Params]): Dictionary mapping each
                child to its values.
            **kwargs: Key word arguments provided by the user.

        Returns:
            Params: The aggregated values.
        """
        return average_state_dicts(children_state_dicts, weights=None)


class FedSGD(Strategy):
    """
    Implementation of the FedSGD strategy, which uses 'FedSGD' for the coordinator
    and aggregators, and defaults for the workers and trainer.
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
