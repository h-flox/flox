from __future__ import annotations

import typing as t
from collections import OrderedDict

from flight.strategies.base import (
    DefaultAggrStrategy,
    DefaultCoordStrategy,
    DefaultTrainerStrategy,
    DefaultWorkerStrategy,
    Strategy,
)

if t.TYPE_CHECKING:
    from flight.federation.topologies.node import NodeID
    from flight.strategies import NodeState, Params


class FedAsyncAggr(DefaultAggrStrategy):
    """The aggregator for 'FedAsync' and its respective methods.

    Args:
        DefaultAggrStrategy: The base class providing necessary methods for FedAsyncAggr.
    """

    def __init__(self, alpha: float = 0.5):
        assert 0.0 < alpha <= 1.0
        self.alpha = alpha

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
            **kwargs: Key Word arguments provided by the user.

        Returns:
            Params: The aggregated values.
        """
        last_updated = kwargs.get("last_updated_node", None)
        assert last_updated is not None
        assert isinstance(last_updated, int | str)

        global_model_params = state.global_model.state_dict()
        last_updated_params = children_state_dicts[last_updated]

        aggr_params = []
        for param in global_model_params:
            w0, w = (
                global_model_params[param].detach(),
                last_updated_params[param].detach(),
            )
            aggr_w = w0 * (1 - self.alpha) + w * self.alpha
            aggr_params.append((param, aggr_w))

        return OrderedDict(aggr_params)


class FedAsync(Strategy):
    """Implementation of the FedAsync strategy, which uses default strategies for coordinator, workers, and trainer
        and the 'FedAsyncAggr'.

    Args:
        Strategy: The base class providing the necessary attributes for 'FedAsync'.
    """

    def __init__(self, alpha: float):
        super().__init__(
            aggr_strategy=FedAsyncAggr(alpha),
            coord_strategy=DefaultCoordStrategy(),
            worker_strategy=DefaultWorkerStrategy(),
            trainer_strategy=DefaultTrainerStrategy(),
        )
