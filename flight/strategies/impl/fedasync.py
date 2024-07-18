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
    def __init__(self, alpha: float):
        super().__init__(
            aggr_strategy=FedAsyncAggr(alpha),
            coord_strategy=DefaultCoordStrategy(),
            worker_strategy=DefaultWorkerStrategy(),
            trainer_strategy=DefaultTrainerStrategy(),
        )
