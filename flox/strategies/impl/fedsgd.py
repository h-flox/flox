import typing as t

from flox.learn.typing import Params
from flox.strategies import Strategy
from flox.strategies.commons.averaging import average_state_dicts
from flox.strategies.commons.worker_selection import random_worker_selection
from flox.strategies.strategy import DefaultClientStrategy, DefaultAggregatorStrategy
from flox.topos import AggrState, NodeID, NodeState


class FedSGDClient(DefaultClientStrategy):
    """
    Federated learning algorithm that does a simple averaging strategy for model aggregation.
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
        self, state: NodeState, workers: t.Iterable[NodeID], seed: int | None = None
    ):
        # print(f"FedSGDClient.select_worker_nodes() :: {workers=}")
        selected_workers = random_worker_selection(
            workers,
            participation=self.participation,
            probabilistic=self.probabilistic,
            always_include_child_aggregators=self.always_include_child_aggregators,
            seed=seed,
        )
        return selected_workers


class FedSGDAggr(DefaultAggregatorStrategy):
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
        """Performs a simple average of the model parameters returned by the child nodes.

        The average is done by:

        $$
            w^{t} \\triangleq \\frac{1}{K} \\sum_{k=1}^{K} w_{k}^{t}
        $$

        where $w^{t}$ is the aggregated model parameters, $K$ is the number of returned
        model updates, $t$ is the current round, and $w_{k}^{t}$ is the returned model
        updates from child $k$ at round $t$.

        Args:
            state (AggrState): ...
            children_states (t.Mapping[NodeID, NodeState]): ...
            children_state_dicts (t.Mapping[NodeID, Params]): ...
            **kwargs: ...

        Returns:
            The averaged parameters.
        """
        return average_state_dicts(children_state_dicts, weights=None)


class FedSGD(Strategy):
    def __init__(
        self,
        participation: float = 1.0,
        probabilistic: bool = False,
        always_include_child_aggregators: bool = True,
    ):
        super().__init__(
            client_strategy=FedSGDClient(
                participation,
                probabilistic,
                always_include_child_aggregators,
            ),
            aggr_strategy=FedSGDAggr(),
        )
