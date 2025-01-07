from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flight.federation.topologies.node import Node, WorkerState
    from flight.learning.types import LocalStepOutput


# TODO: Remove this 'feature' from the Strategy class. For now, it is not needed.
#       Instead, focus your time on the Globus Compute and Parsl implementations to
#       run FL with basic testing/validation.


@t.runtime_checkable
class TrainerStrategy(t.Protocol):
    """
    Protocol for all trainer strategies, including those defined in
    Flight and those defined by users.
    """

    # noinspection PyMethodMayBeStatic
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
            node (Node | None): The `Node` hyperparameters are loaded on. This can be
                used for an implementation of this method to use node-specific logic
                or take advantage of the node's `extra` cache to inform hyperparameters.
            state (WorkerState | None): The state of the current worker node.

        Returns:
            Trainer hyperparameters. What key-value pairs are returned will need to
            depend on the trainer users plan to use for federations. The default
            behavior of this method is to return an empty dictionary.
        """
        return {}

    # noinspection PyMethodMayBeStatic
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
        return out

    # noinspection PyMethodMayBeStatic
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
        return out
