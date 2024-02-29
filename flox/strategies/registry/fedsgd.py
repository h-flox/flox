from __future__ import annotations

import typing

from flox.flock import FlockNode, FlockNodeID
from flox.flock.states import AggrState, NodeState
from flox.strategies.base import Strategy
from flox.strategies.commons.averaging import average_state_dicts
from flox.strategies.commons.worker_selection import random_worker_selection

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from flox.nn.typing import StateDict


class FedSGD(Strategy):
    """Implementation of the Federated Stochastic Gradient Descent algorithm.

    In short, this algorithm randomly selects a subset of worker nodes and will
    do a simple, unweighted average across the updates to the model parameters
    (i.e., ``StateDict``).

    > **Reference:**
    >
    > McMahan, Brendan, et al. "Communication-efficient learning of deep networks
    > from decentralized data." Artificial intelligence and statistics. PMLR, 2017.
    """

    def __init__(
        self,
        participation: float = 1.0,
        probabilistic: bool = True,
        always_include_child_aggregators: bool = True,
        seed: int | None = None,
    ):
        """Initializes the FedSGD strategy with the desired parameters.

        Args:
            participation (float): Fraction of the child nodes to be selected.
            probabilistic (bool): If `True`, nodes are selected entirely probabilistically rather than
                based on a fraction (`False`). As an example, consider you have 10 children nodes to select from and
                `participation=0.5`. If `probabilistic=True`, then exactly 5 children nodes *will* be selected;
                otherwise, then each child node will be selected with probability 0.5.
            always_include_child_aggregators (bool): If `True`, child aggregator nodes will always be included;
                if `False`, then they will only be included if they are naturally selected (similar to worker
                child nodes).
            seed (int): Random seed. # TODO: Change this to standardized seeding format.
        """
        super().__init__()
        assert 0.0 <= participation <= 1.0
        self.participation = participation
        self.probabilistic = probabilistic
        self.always_include_child_aggregators = always_include_child_aggregators
        self.seed = seed

    def agg_worker_selection(
        self,
        state: AggrState,
        children: Iterable[FlockNode],
        **kwargs,
    ) -> list[FlockNode]:
        """Performs a simple average of the model weights returned by the child nodes.

        The average is done by:

        $$
            w^{t} \\triangleq \\frac{1}{K} \\sum_{k=1}^{K} w_{k}^{t}
        $$

        where $w^{t}$ is the aggregated model weights, $K$ is the number of returned
        model updates, $t$ is the current round, and $w_{k}^{t}$ is the returned model
        updates from child $k$ at round $t$.

        Args:
            state (AggrState): ...
            children (list[FlockNode]): ...
            **kwargs: ...

        Returns:
            The selected children nodes.
        """
        return random_worker_selection(
            children,
            participation=self.participation,
            probabilistic=self.probabilistic,
            always_include_child_aggregators=self.always_include_child_aggregators,
            seed=self.seed,  # TODO: Change this because it will always do the same thing as is.
        )

    def agg_param_aggregation(
        self,
        state: AggrState,
        children_states: Mapping[FlockNodeID, NodeState],
        children_state_dicts: Mapping[FlockNodeID, StateDict],
        **kwargs,
    ) -> StateDict:
        """Runs simple, unweighted averaging of ``StateDict`` objects from each child node.

        Args:
            state (AggrState): ...
            children_states (dict[FlockNodeID, NodeState]): ...
            children_state_dicts (dict[FlockNodeID, StateDict]): ...
            *args: ...
            **kwargs: ...

        Returns:
            The averaged ``StateDict``.
        """
        return average_state_dicts(children_state_dicts, weights=None)
