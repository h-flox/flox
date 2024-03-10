from __future__ import annotations

import typing as t

from flox.strategies.commons.averaging import average_state_dicts

if t.TYPE_CHECKING:
    from flox.flock import AggrState, NodeID, NodeState
    from flox.nn.typing import Params


class AggregatorStrategy(t.Protocol):
    def round_start(self):
        _ = 0

    def aggregate_params(
        self,
        state: AggrState,
        children_states: t.Mapping[NodeID, NodeState],
        children_state_dicts: t.Mapping[NodeID, Params],
        **kwargs,
    ) -> Params:
        return average_state_dicts(children_state_dicts, weights=None)

    def round_end(self):
        _ = 0
