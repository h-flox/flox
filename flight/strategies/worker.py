from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    import torch

    from flight.federation.jobs.result import Result
    from flight.federation.topologies.node import WorkerState


@t.runtime_checkable
class WorkerStrategy(t.Protocol):
    """
    Template for all aggregator strategies, including those defined in
    Flight and those defined by users.
    """

    def start_work(self, state: WorkerState) -> WorkerState:
        """Callback to run at the start of the current nodes 'work'.

        Args:
            state (WorkerState): State of the current worker node.

        Returns:
            The state of the current node at the end of the callback.
        """

    def before_training(
        self,
        state: WorkerState,
        data: t.Any,
    ) -> tuple[WorkerState, t.Any]:
        """Callback to run before the current nodes training.

        Args:
            state (WorkerState): State of the current worker node.
            data (t.Any): The data related to the current worker node.

        Returns:
            A tuple containing the state and data of the current worker node
            after the callback.
        """

    def after_training(
        self,
        state: WorkerState,
        optimizer: torch.optim.Optimizer,
    ) -> WorkerState:
        """Callback to run after the current nodes training.

        Args:
            state (WorkerState): State of the current worker node.
            optimizer (torch.optim.Optimizer): The PyTorch optimizer used.

        Returns:
            The state of the current worker node after the callback.
        """

    def end_work(self, result: Result) -> Result:
        """Callback to run at the end of the current worker nodes 'work'

        Args:
            result (Result): A Result object used to represent the result of the
                local training on the current worker node.

        Returns:
            The result of the worker nodes local training.
        """
