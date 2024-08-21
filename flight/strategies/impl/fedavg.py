from __future__ import annotations

import typing as t

from ..base import (
    DefaultAggrStrategy,
    DefaultTrainerStrategy,
    DefaultWorkerStrategy,
    Strategy,
)
from ..commons.averaging import average_state_dicts
from .fedsgd import FedSGDCoord

if t.TYPE_CHECKING:
    from flight.federation.topologies.node import (
        AggrState,
        NodeID,
        NodeState,
        WorkerState,
    )
    from flight.learning.modules.prototypes import DataModuleProto
    from flight.learning.types import Params


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
        children_params: t.Mapping[NodeID, Params],
        **kwargs,
    ) -> Params:
        """
        Method used by aggregator nodes for aggregating the passed node state
        dictionary.

        Args:
            state (NodeState): State of the current aggregator node.
            children_states (t.Mapping[NodeID, NodeState]): Dictionary of the states
                of the children.
            children_params (t.Mapping[NodeID, Params]): Dictionary mapping each
                child to its values.
            **kwargs: Key word arguments provided by the user.

        Returns:
            Params: The aggregated values.
        """
        weights = {}
        for node, child_state in children_states.items():
            weights[node] = child_state[FedAvgAggr.NUM_SAMPLES]

        state[FedAvgAggr.NUM_SAMPLES] = sum(weights.values())
        return average_state_dicts(children_params, weights=weights)


class FedAvgWorker(DefaultWorkerStrategy, _FedAvgConstMixins):
    """The worker for 'FedAvg' and its respective methods."""

    def before_training(
        self, state: WorkerState, data: DataModuleProto
    ) -> tuple[WorkerState, DataModuleProto]:
        """Callback to run before the current nodes training.

        Args:
            state (WorkerState): State of the current worker node.
            data (Params): The data related to the current worker node.

        Returns:
            tuple[WorkerState, Params]: A tuple containing the updated state of the
                worker node and the data.
        """
        try:
            state[FedAvgWorker.NUM_SAMPLES] = data.size(state.idx)  # noqa
            return state, data
        except TypeError:
            raise TypeError(
                "FedAvgWorker.before_training(): Given `DataLoadable` has not "
                "overridden `__len__`, which is a requirement for the `FedAvg` "
                "strategy."
            )


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
            aggr_strategy=FedAvgAggr(),
            worker_strategy=FedAvgWorker(),
            trainer_strategy=DefaultTrainerStrategy(),
        )
