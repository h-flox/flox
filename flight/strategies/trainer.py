from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flight.federation.topologies.node import Node, WorkerState
    from flight.learning.types import LocalStepOutput


@t.runtime_checkable
class TrainerStrategy(t.Protocol):
    """
    Protocol for all trainer strategies, including those defined in
    Flight and those defined by users.
    """

    def trainer_hparams(
        self,
        node: Node | None = None,
        state: WorkerState | None = None,
    ) -> dict[str, t.Any]:
        """..."""

    def before_backprop(
        self,
        state: WorkerState,
        out: LocalStepOutput,
    ) -> LocalStepOutput:
        """Callback to run before backpropagation.

        Args:
            state (WorkerState): State of the current node.
            out (LocalStepOutput): The calculated loss

        Returns:
            The loss at the end of the callback
        """
        pass

    def after_backprop(
        self,
        state: WorkerState,
        out: LocalStepOutput,
    ) -> LocalStepOutput:
        """Callback to run after backpropagation.

        Args:
            state (WorkerState): State of the current node.
            out (LocalStepOutput): The calculated loss

        Returns:
            The loss at the end of the callback
        """
        pass
