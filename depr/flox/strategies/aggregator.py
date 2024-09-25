from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flox.federation.topologies import AggrState, NodeID, NodeState
    from flox.learn.types import Params


class AggregatorStrategy(t.Protocol):
    def round_start(self):
        """Callback to run at the *start* of a round."""
        pass

    def aggregate_params(
        self,
        state: AggrState,
        children_states: t.Mapping[NodeID, NodeState],
        children_state_dicts: t.Mapping[NodeID, Params],
        **kwargs,
    ) -> Params:
        """Callback that handles the model parameter aggregation step.

        Args:
            state (AggrState): This aggregator node's current state.
            children_states (t.Mapping[NodeID, NodeState]): The states of children nodes.
            children_state_dicts (t.Mapping[NodeID, Params]): The model parameters of models owned by children nodes.
            **kwargs: Keyword arguments provided by users.

        Returns:
            The aggregated parameters to update the model at the respective aggregator.
        """
        pass

    def round_end(self):
        """Callback to run at the *end* of a round."""
        pass
