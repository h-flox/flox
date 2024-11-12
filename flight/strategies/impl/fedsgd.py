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
from flight.strategies.commons import random_worker_selection

if t.TYPE_CHECKING:
    from flight.federation.topologies.node import Node, NodeState


class FedSGDCoord(DefaultCoordStrategy):
    """
    The coordinator and its respective methods for 'FedSGD'.
    """

    def __init__(
        self,
        participation: float = 1.0,
        probabilistic: bool = False,
        always_include_child_aggregators: bool = True,
    ):
        """

        Args:
            participation (float): The proportion of *all* worker nodes in the topology
                that will participate in a given federation round.
            probabilistic (bool): Whether the selection of nodes will be probabilistic.
                If `True`, then each worker node will be selected with probability
                `participation`; if `False` then a fixed set of $n$ nodes will be
                selected with where $n = \\max(1, |W| \\cdot \\texttt{participation})$
                where $|W|$ is the number of workers in the federation's topology.
            always_include_child_aggregators:
        """
        self.participation = participation
        self.probabilistic = probabilistic
        self.always_include_child_aggregators = always_include_child_aggregators

    def select_workers(
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
            Worker nodes selected to participate in a federation round.
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

    '''
    def aggregate_params(
        self,
        state: AggrState,
        children_states: t.Mapping[NodeID, NodeState],
        children_modules: t.Mapping[NodeID, AbstractModule],
        **kwargs: dict[str, t.Any],
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
            children_modules (t.Mapping[NodeID, AbstractModule]): A mapping of the
                modules belonging to each respective child node.
            **kwargs: Key word arguments provided by the user.

        Returns:
            Aggregated parameters.
        """
        print(children_modules)
        children_params = {
            child: module.get_params(to_numpy=True)
            for child, module in children_modules.items()
        }
        return average_state_dicts(children_params, weights=None)
    '''


class FedSGD(Strategy):
    """
    Implementation of the FedSGD strategy, which uses 'FedSGD' for the coordinator
    and aggregators, and defaults for the workers and trainer.

    References:
        McMahan, Brendan, et al. "Communication-efficient learning of deep networks
        from decentralized data." *Artificial intelligence and statistics*. PMLR, 2017.
    """

    def __init__(
        self,
        participation: float = 1.0,
        probabilistic: bool = False,
        always_include_child_aggregators: bool = True,
    ):
        """

        Args:
            participation (float): The proportion of *all* worker nodes in the topology
                that will participate in a given federation round.
            probabilistic (bool): Whether the selection of nodes will be probabilistic.
                If `True`, then each worker node will be selected with probability
                `participation`; if `False` then a fixed set of $n$ nodes will be
                selected with where $n = \\max(1, |W| \\cdot \\texttt{participation})$
                where $|W|$ is the number of workers in the federation's topology.
            always_include_child_aggregators (bool):
        """
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
