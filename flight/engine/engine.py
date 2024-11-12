from __future__ import annotations

import typing as t
from concurrent.futures import Future

from .control.serial import SerialController
from .data.base import AbstractTransfer, BaseTransfer

if t.TYPE_CHECKING:
    from ..types import P, T
    from .control.base import AbstractController


class Engine:
    """
    A wrapper object for the controller and transmitter objects that are used to
    facilitate and execute a federation in Flight.
    """

    controller: AbstractController
    """
    Object responsible for submitting functions to be executed at the appropriate
    compute resources (e.g., compute nodes, threads, processes).
    """

    transmitter: AbstractTransfer
    """
    Object responsible for facilitating data transfer for the execution of jobs.
    This abstraction is used in the case of distributed and remote execution
    of Flight federations.
    """

    def __init__(
        self,
        controller: AbstractController,
        transmitter: AbstractTransfer,
    ):
        """
        Initializes the engine with the given controller and transmitter.

        Args:
            controller (AbstractController): The controller responsible for submitting
                functions to be executed at the appropriate compute resources.
            transmitter (AbstractTransfer): The object responsible for facilitating data
                transfers for the execution of jobs.
        """
        self.controller = SerialController()
        self.transmitter = BaseTransfer()

    def submit(self, fn: t.Callable, **kwargs: dict[str, t.Any]) -> Future:
        """
        Submits a function to be executed by the controller.

        Args:
            fn (t.Callable): The function to be executed.
            **kwargs (dict[str, t.Any]): Keyword arguments to be passed to the function.

        Returns:
            A future object representing the asynchronous execution of the function.
        """
        return self.controller.submit(fn, **kwargs)

    def transfer(self, data: P) -> T:
        """
        Transfers data using the transmitter object.

        Args:
            data: The data to transfer.

        Returns:
            A reference to the returned object. In the local execution case, this will
            likely be the original data itself. However, this could be a reference to
            another object that will finish the transfer in a JIT way.
        """
        return self.transmitter.transfer(data)

    @classmethod
    def setup(
        cls,
        controller_kind: ...,
        transmitter_kind: ...,
        controller_cfg: dict[str, t.Any] | None = None,
        transmitter_cfg: dict[str, t.Any] | None = None,
    ) -> Engine:
        """
        This helper method prepares a new `Engine` instance.

        Args:
            controller_kind: ...
            transmitter_kind: ...
            controller_cfg (dict[str, t.Any]): ...
            transmitter_cfg (dict[str, t.Any]): ...

        Returns:
            An `Engine` instance based on the provided configurations.
        """
        # TODO
        controller: AbstractController = None
        transmitter: AbstractTransfer = None
        return cls(controller, transmitter)
