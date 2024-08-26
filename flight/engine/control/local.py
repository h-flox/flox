from __future__ import annotations

import typing as t
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

from .base import AbstractController

if t.TYPE_CHECKING:
    from concurrent.futures import Executor


class LocalController(AbstractController):
    """
    A local controller (similar to
    [`SerialController`][flight.engine.control.serial.SerialController]) that instead
    runs multiple functions at once using either threads or processes.
    """

    executor: Executor

    def __init__(
        self,
        kind: str | t.Literal["process", "thread"],
        **executor_kwargs,
    ):
        self.kind = kind
        match self.kind:
            case "process":
                self.executor = ProcessPoolExecutor(**executor_kwargs)
            case "thread":
                self.executor = ThreadPoolExecutor(**executor_kwargs)

    def __call__(self, fn: t.Callable, /, **kwargs) -> Future:
        return self.executor.submit(fn, **kwargs)

    def shutdown(self):
        self.executor.shutdown(wait=True, cancel_futures=False)
