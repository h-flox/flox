from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    import torch

    from flight.federation.jobs.result import Result
    from flight.strategies import NodeState


@t.runtime_checkable
class WorkerStrategy(t.Protocol):
    """Template for all aggregator strategies, including those defined in Flight and those defined by Users."""

    def start_work(self, state: NodeState) -> NodeState:
        """Callback to run at the start of the current nodes 'work'.

        Args:
            state (NodeState): State of the current worker node.

        Returns:
            The state of the current node at the end of the callback.
        """
        pass

    def before_training(self, state: NodeState, data: t.Any) -> tuple[NodeState, t.Any]:
        """Callback to run before the current nodes training.

        Args:
            state (NodeState): State of the current worker node.
            data (t.Any): The data related to the current worker node.

        Returns:
            A tuple containing the state and data of the current worker node after the callback.
        """
        pass

    def after_training(
        self, state: NodeState, optimizer: torch.optim.Optimizer
    ) -> NodeState:
        """Callback to run after the current nodes training.

        Args:
            state (NodeState): State of the current worker node.
            optimizer (torch.optim.Optimizer): The PyTorch optimizer used.

        Returns:
            The state of the current worker node after the callback.
        """
        pass

    def end_work(self, result: Result) -> Result:
        """Callback to run at the end of the current worker nodes 'work'

        Args:
            result (Result): A Result object used to represent the result of the local training on the current worker node.

        Returns:
            The result of the worker nodes local training.

        """
        pass
