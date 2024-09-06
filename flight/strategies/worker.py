from __future__ import annotations

import typing as t

from flight.learning.base import AbstractDataModule

if t.TYPE_CHECKING:
    import torch

    from flight.federation.jobs.types import Result
    from flight.federation.topologies.node import WorkerState


@t.runtime_checkable
class WorkerStrategy(t.Protocol):
    """
    Template for all aggregator strategies, including those defined in
    Flight and those defined by users.
    """

    # noinspection PyMethodMayBeStatic
    def start_work(self, state: WorkerState) -> WorkerState:
        """Callback to run at the start of the current nodes 'work'.

        Args:
            state (WorkerState): State of the current worker node.

        Returns:
            The state of the current node at the end of the callback.
        """
        return state

    # noinspection PyMethodMayBeStatic
    def before_training(
        self,
        state: WorkerState,
        data: AbstractDataModule,
    ) -> tuple[WorkerState, AbstractDataModule]:
        """Callback to run before the current nodes training.

        Args:
            state (WorkerState): State of the current worker node.
            data (AbstractDataModule): The data module for loading the data available
                on the current worker node.

        Returns:
            A tuple containing the following:

                - the worker's state
                - data of the current worker node

                These are returned at the end of the callback in the case that the user
                implements logic that modifies them before training.
        """
        return state, data

    # noinspection PyMethodMayBeStatic
    def after_training(
        self,
        state: WorkerState,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[WorkerState, torch.optim.Optimizer]:
        """Callback to run after the current nodes training.

        Args:
            state (WorkerState): State of the current worker node.
            optimizer (torch.optim.Optimizer): The PyTorch optimizer used.

        Returns:
            The state of the current worker node after the callback.
        """
        return state, optimizer

    # noinspection PyMethodMayBeStatic
    def end_work(self, result: Result) -> Result:
        """Callback to run at the end of the current worker nodes 'work'

        Args:
            result (Result): A Result object used to represent the result of the
                local training on the current worker node.

        Returns:
            The result of the worker nodes local training.
        """
        return result
