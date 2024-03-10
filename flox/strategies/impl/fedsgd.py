import typing as t

from flox.flock import NodeID, NodeState, AggrState
from flox.nn.typing import Params
from flox.strategies import Strategy, AggregatorStrategy, ClientStrategy
from flox.strategies.commons.averaging import average_state_dicts
from flox.strategies.commons.worker_selection import random_worker_selection


class FedSGDClient(ClientStrategy):
    """
    ...
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

    def select_worker_nodes(self, state, children, seed):
        return random_worker_selection(
            children,
            participation=self.participation,
            probabilistic=self.probabilistic,
            always_include_child_aggregators=self.always_include_child_aggregators,
            seed=seed,
        )


class FedSGDAggr(AggregatorStrategy):
    """
    ...
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
