from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flight.federation.topologies.node import NodeID
    from flight.strategies import NodeState, Params


@t.runtime_checkable
class AggrStrategy(t.Protocol):
    """Template for all aggregator strategies, including those defined in Flight and those defined by Users."""

    def start_round(self):
        """Callback to run at the start of a round."""
        pass

    def aggregate_params(
        self,
        state: NodeState,
        children_states: t.Mapping[NodeID, NodeState],
        children_state_dicts: t.Mapping[NodeID, Params],
        **kwargs,
    ) -> Params:
        """Callback that handles the model parameter aggregation step.

        Args:
            state (NodeState): The state of the current aggregator node.
            children_states (t.Mapping[NodeID, NodeState]): A mapping of the current aggregator node's children and their respective states.
            children_state_dicts (t.Mapping[NodeID, Parmas]): The model parameters of the models to each respective child node.
            **kwargs: Keyword arguments provided by users.

        Returns:
            Params: The aggregated parameters to update the model at the current aggregator.
        """
        pass

    def end_round(self):
        """Callback to run at the end of a round."""
        pass
