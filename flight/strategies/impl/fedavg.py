from __future__ import annotations

import typing as t

from flight.strategies.base import (
    DefaultAggrStrategy,
    DefaultTrainerStrategy,
    DefaultWorkerStrategy,
    Strategy,
)
from flight.strategies.commons import average_state_dicts

if t.TYPE_CHECKING:
    from flight.federation.topologies.node import (
        AggrState,
        NodeID,
        NodeState,
        WorkerState,
    )

    from ...learning.types import Params
    from .fedsgd import FedSGDCoord


class _FedAvgConstMixins:
    """Defines common constants throughout that will be used for `FedAvg` classes."""

    NUM_SAMPLES = "num_data_samples"


class FedAvgAggr(DefaultAggrStrategy, _FedAvgConstMixins):
    """
    Performs a weighted average of the model parameters returned by the child nodes.

    The average is done by:

    $$
        w^{t} \\triangleq \\sum_{k=1}^{K} \\frac{n_{k}}{n} w_{k}^{t}
    $$

    where $n_{k}$ is the number of data items at worker $k$
    (and $n \\triangleq \\sum_{k} n_{k}$), $w^{t}$ is the aggregated model parameters,
    $K$ is the number of returned model updates, $t$ is the current round, and
    $w_{k}^{t}$ is the returned model updates from child $k$ at round $t$.
    """

    def aggregate_params(
        self,
        state: AggrState,
        children_states: t.Mapping[NodeID, NodeState],
        children_state_dicts: t.Mapping[NodeID, Params],
        **kwargs,
    ) -> Params:
        """
        Method used by aggregator nodes for aggregating the passed node state
        dictionary.

        Args:
            state (NodeState): State of the current aggregator node.
            children_states (t.Mapping[NodeID, NodeState]): Dictionary of the states
                of the children.
            children_state_dicts (t.Mapping[NodeID, Params]): Dictionary mapping each
                child to its values.
            **kwargs: Key word arguments provided by the user.

        Returns:
            Params: The aggregated values.
        """
        weights = {}
        for node, child_state in children_states.items():
            weights[node] = child_state[FedAvgAggr.NUM_SAMPLES]

        state[FedAvgAggr.NUM_SAMPLES] = sum(weights.values())
        return average_state_dicts(children_state_dicts, weights=weights)


class FedAvgWorker(DefaultWorkerStrategy, _FedAvgConstMixins):
    """The worker for 'FedAvg' and its respective methods."""

    def before_training(
        self, state: WorkerState, data: Params
    ) -> tuple[WorkerState, Params]:
        """Callback to run before the current nodes training.

        Args:
            state (WorkerState): State of the current worker node.
            data (Params): The data related to the current worker node.

        Returns:
            tuple[WorkerState, Params]: A tuple containing the updated state of the
                worker node and the data.
        """
        state[FedAvgWorker.NUM_SAMPLES] = len(data)
        return state, data


class FedAvg(Strategy):
    """
    Implementation of the Federated Averaging algorithm.

    This algorithm extends ``FedSGD`` and differs from it by performing a weighted
    average based on the number of data samples each (sibling) worker has. Worker
    selection is done randomly, same as ``FedSGD``.

    References:
        McMahan, Brendan, et al. "Communication-efficient learning of deep networks
        from decentralized data." *Artificial intelligence and statistics*. PMLR, 2017.
    """

    def __init__(
        self,
        participation: float = 1.0,
        probabilistic: bool = False,
        always_include_child_aggregators: bool = False,
    ):
        """

        Args:
            participation:
            probabilistic:
            always_include_child_aggregators:
        """
        super().__init__(
            coord_strategy=FedSGDCoord(
                participation,
                probabilistic,
                always_include_child_aggregators,
            ),
            aggr_strategy=FedAvgAggr(),
            worker_strategy=FedAvgWorker(),
            trainer_strategy=DefaultTrainerStrategy(),
        )
