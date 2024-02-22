from flox.flock import FlockNodeID
from flox.flock.states import FloxWorkerState
from flox.flock.states import NodeState, FloxAggregatorState
from flox.strategies.commons.averaging import average_state_dicts
from flox.strategies.registry.fedsgd import FedSGD
from flox.typing import StateDict


class FedAvg(FedSGD):
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
        probabilistic: bool = True,
        always_include_child_aggregators: bool = True,
        seed: int | None = None,
    ):
        """

        Args:
            participation (float): Participation rate for random worker selection.
            probabilistic (bool): Probabilistically chooses workers if True; otherwise will always
                select `max(1, max_workers * participation)` workers.
            always_include_child_aggregators (bool): If True, Will always include child nodes that are
                aggregators; if False, then they are included at random.
            seed (int): Random seed.
        """
        super().__init__(
            participation, probabilistic, always_include_child_aggregators, seed
        )

    def wrk_before_train_step(self, state: FloxWorkerState, *args, **kwargs):
        if "dataset" not in kwargs:
            raise ValueError("`dataset` must be provided")
        state["num_data_samples"] = len(kwargs["dataset"])

    def agg_param_aggregation(
        self,
        state: FloxAggregatorState,
        children_states: dict[FlockNodeID, NodeState],
        children_state_dicts: dict[FlockNodeID, StateDict],
        *args,
        **kwargs,
    ):
        weights = {}
        for node, child_state in children_states.items():
            weights[node] = child_state["num_data_samples"]
        state["num_data_samples"] = sum(weights.values())
        return average_state_dicts(children_state_dicts, weights=weights)
