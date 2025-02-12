from __future__ import annotations

import typing as t
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

from .base import AbstractController

if t.TYPE_CHECKING:
    from concurrent.futures import Executor

    from flight.types import P, T


class LocalController(AbstractController):
    """
    A local controller (similar to
    [`SerialController`][flight.engine.controllers.serial.SerialController]) that
    instead runs multiple functions at once using either threads or processes.
    """

    executor: Executor

    def __init__(
        self,
        kind: str | t.Literal["process", "thread"],
        **executor_kwargs,
    ):
        """
        Initializes the controller with the given kind of executor.

        Args:
            kind (str | t.Literal["process", "thread"]): The kind of executor to use.
            **executor_kwargs: Keyword arguments to be passed to the executor.
        """
        self.kind = kind
        match self.kind:
            case "process":
                self.executor = ProcessPoolExecutor(**executor_kwargs)
            case "thread":
                self.executor = ThreadPoolExecutor(**executor_kwargs)

    def submit(self, fn: t.Callable[P, T], /, **kwargs) -> Future[T]:  # noqa
        """
        Executes a function locally and asynchronously using either threads or
        processes based on the given arguments to `__init__`.

        Args:
            fn (t.Callable[P, T]): The function to be executed.
            **kwargs: Keyword arguments to be passed to the function.

        Returns:
            A future object representing the asynchronous execution of the function.

        Notes:
            Arguments to function can *only* be passed via keyword arguments.
        """
        return self.executor.submit(fn, **kwargs)

    def shutdown(self):
        """
        Shuts down the executor.
        """
        self.executor.shutdown(wait=True, cancel_futures=False)
