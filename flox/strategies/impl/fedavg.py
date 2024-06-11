from __future__ import annotations

import typing as t

from flox.strategies import Strategy
from flox.strategies.commons.averaging import average_state_dicts
from flox.strategies.impl.fedsgd import FedSGDClient
from flox.strategies.strategy import DefaultAggregatorStrategy, DefaultWorkerStrategy

if t.TYPE_CHECKING:
    from flox.topos import AggrState, NodeID, NodeState, WorkerState
    from flox.learn.typing import Params


class FedAvgAggr(DefaultAggregatorStrategy):
    def aggregate_params(
        self,
        state: AggrState,
        children_states: t.Mapping[NodeID, NodeState],
        children_state_dicts: t.Mapping[NodeID, Params],
        **kwargs,
    ) -> Params:
        """Performs a weighted average of the model parameters returned by the child nodes.

        The average is done by:

        $$
            w^{t} \\triangleq \\sum_{k=1}^{K} \\frac{n_{k}}{n} w_{k}^{t}
        $$

        where $n_{k}$ is the number of data items at worker $k$ (and $n \\triangleq \\sum_{k} n_{k}$),
        $w^{t}$ is the aggregated model parameters, $K$ is the number of returned
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
        weights = {}
        for node, child_state in children_states.items():
            weights[node] = child_state["num_data_samples"]
        state["num_data_samples"] = sum(weights.values())

        # avg_samples = state["num_data_samples"] / len(children_states)
        # remaining_children = len(list(state.children)) - len(children_states)
        # state["_num_data_samples"] = (
        #     state["num_data_samples"] + avg_samples * remaining_children
        # )
        # for node in state.children:
        #     if node not in children_states:
        #         weights[node.idx] = avg_samples
        #         children_state_dicts[node.idx] = state.global_model.state_dict()

        return average_state_dicts(children_state_dicts, weights=weights)


class FedAvgWorker(DefaultWorkerStrategy):
    def before_training(
        self, state: WorkerState, data: t.Any
    ) -> tuple[WorkerState, t.Any]:
        state["num_data_samples"] = len(data)
        return state, data


class FedAvg(Strategy):
    """Implementation of the Federated Averaging algorithm.

    This algorithm extends ``FedSGD`` and differs from it by performing a weighted
    average based on the number of data samples each (sibling) worker has. Worker
    selection is done randomly, same as ``FedSGD``.

    References:
        McMahan, Brendan, et al. "Communication-efficient learning of deep networks
        from decentralized data." Artificial intelligence and statistics. PMLR, 2017.
    """

    def __init__(
        self,
        participation: float = 1.0,
        probabilistic: bool = False,
        always_include_child_aggregators: bool = False,
    ):
        super().__init__(
            client_strategy=FedSGDClient(
                participation,
                probabilistic,
                always_include_child_aggregators,
            ),
            aggr_strategy=FedAvgAggr(),
            worker_strategy=FedAvgWorker(),
        )
