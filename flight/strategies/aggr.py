from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flight.federation.topologies.node import AggrState, NodeID, NodeState
    from flight.learning.types import Params


@t.runtime_checkable
class AggrStrategy(t.Protocol):
    """
    Template for all aggregator strategies, including those defined in
    Flight and those defined by users.
    """

    def start_round(self):
        """
        Callback to run at the start of a round.
        """

    def aggregate_params(
        self,
        state: AggrState,
        children_states: t.Mapping[NodeID, NodeState],
        children_state_dicts: t.Mapping[NodeID, Params],
        **kwargs,
    ) -> Params:
        """Callback that handles the model parameter aggregation step.

        Args:
            state (AggrState): The state of the current aggregator node.
            children_states (t.Mapping[NodeID, NodeState]): A mapping of the current
                aggregator node's children and their respective states.
            children_state_dicts (t.Mapping[NodeID, Params]): The model parameters of
                the models to each respective child node.
            **kwargs: Keyword arguments provided by users.

        Returns:
            The aggregated parameters to update the model at the current aggregator.
        """

    def end_round(self):
        """
        Callback to run at the end of a round.
        """
