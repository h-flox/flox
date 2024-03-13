from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flox.flock import AggrState, NodeID, NodeState
    from flox.nn.typing import Params


class AggregatorStrategy(t.Protocol):
    def round_start(self):
        pass

    def aggregate_params(
        self,
        state: AggrState,
        children_states: t.Mapping[NodeID, NodeState],
        children_state_dicts: t.Mapping[NodeID, Params],
        **kwargs,
    ) -> Params:
        pass

    def round_end(self):
        pass
