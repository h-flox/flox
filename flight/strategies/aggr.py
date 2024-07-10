from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    Params: t.TypeAlias = t.Any
    NodeState: t.TypeAlias = t.Any
    NodeID: t.TypeAlias = t.Any


@t.runtime_checkable
class AggrStrategy(t.Protocol):
    def start_round(self):
        pass

    def aggregate_params(
        self,
        state: NodeState,
        children_states: t.Mapping[NodeID, NodeState],
        children_state_dicts: t.Mapping[NodeID, Params],
        **kwargs,
    ) -> Params:
        pass

    def end_round(self):
        pass
