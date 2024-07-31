from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flight.strategies import Loss, NodeState


@t.runtime_checkable
class TrainerStrategy(t.Protocol):
    """Template for all trainer strategies, including those defined in Flight and those defined by users."""

    def before_backprop(self, state: NodeState, loss: Loss) -> Loss:
        """Callback to run before backpropagation.

        Args:
            state (NodeState): State of the current node.
            loss (Loss): The calculated loss

        Returns:
            The loss at the end of the callback
        """
        pass

    def after_backprop(self, state: NodeState, loss: Loss) -> Loss:
        """Callback to run after backpropagation.

        Args:
            state (NodeState): State of the current node.
            loss (Loss): The calculated loss

        Returns:
            The loss at the end of the callback
        """
        pass
