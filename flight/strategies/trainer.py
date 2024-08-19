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

    def hparams(
        self,
        node: Node | None = None,
        state: WorkerState | None = None,
    ) -> dict[str, t.Any]:
        """
        Returns the hyperparameters to be used by the ``Trainer`` for local training.

        This can be defined by the user's ``TrainerStrategy`` implementation to take
        advantage of node-specific data. For instance, if a node has a value cached
        in its `node.extra` cache, then you can implement logic to incorporate those
        values for smarter, more complex FL heuristics/algorithms.

        Args:
            node:
            state:

        Returns:

        """

    def before_backprop(
        self,
        state: WorkerState,
        out: LocalStepOutput,
    ) -> LocalStepOutput:
        """
        Callback to run before backpropagation.

        Args:
            state (WorkerState): State of the current node.
            out (LocalStepOutput): The calculated loss

        Returns:
            Loss after running the callback.
        """

    def after_backprop(
        self,
        state: WorkerState,
        out: LocalStepOutput,
    ) -> LocalStepOutput:
        """
        Callback to run after backpropagation.

        Args:
            state (WorkerState): State of the current node.
            out (LocalStepOutput): The calculated loss

        Returns:
            Loss after running the callback.
        """
