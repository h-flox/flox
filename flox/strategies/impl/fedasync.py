from __future__ import annotations

import typing as t
from collections import OrderedDict

from flox.flock import AggrState, NodeID, NodeState
from flox.nn.typing import Params
from flox.strategies import Strategy
from flox.strategies.strategy import DefaultAggregatorStrategy


class FedAsyncAggr(DefaultAggregatorStrategy):
    def __init__(self, alpha: float = 0.5):
        assert 0.0 < alpha <= 1.0
        self.alpha = alpha

    def aggregate_params(
        self,
        state: AggrState,
        children_states: t.Mapping[NodeID, NodeState],
        children_state_dicts: t.Mapping[NodeID, Params],
        **kwargs,
    ) -> Params:
        last_updated = kwargs.get("last_updated_node", None)
        if last_updated is None:
            raise ValueError

        global_model_params = state.global_model.state_dict()
        last_updated_params = children_state_dicts[last_updated]

        aggr_params = []
        print(len(global_model_params), len(last_updated_params))
        for param in global_model_params:
            print(param)
            w0, w = (
                global_model_params[param].detach(),
                last_updated_params[param].detach(),
            )
            aggr_w = w0 * (1 - self.alpha) + w * self.alpha
            aggr_params.append((param, aggr_w))

        return OrderedDict(aggr_params)


class FedAsync(Strategy):
    def __init__(self, alpha: float):
        super().__init__(aggr_strategy=FedAsyncAggr(alpha))
