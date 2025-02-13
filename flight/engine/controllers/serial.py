from __future__ import annotations

import typing as t
from concurrent.futures import Future

from .base import AbstractController

if t.TYPE_CHECKING:
    from flight.types import P, T


class SerialController(AbstractController):
    """
    A simple controller that runs aggr serially.

    This class should be used for rapid prototyping and simple debugging on
    your local machine.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def submit(self, fn: t.Callable[P, T], /, **kwargs) -> Future[T]:  # noqa
        """
        Executes the given function with the provided keyword arguments in a serial
        manner (i.e., no asynchronous or parallel/concurrent execution).

        Args:
            fn (t.Callable[P, T]): The function to be executed.
            **kwargs: Keyword arguments to be passed to the function.

        Returns:
            A future object representing the execution of the function.
        """
        future: Future = Future()
        try:
            result = fn(**kwargs)
            future.set_result(result)
        except Exception as exception:
            future.set_exception(exception)
        return future

    def shutdown(self):
        """This function does nothing for this controller."""
        return None
