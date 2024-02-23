from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, Protocol

from flox.flock import FlockNode


class LauncherFunction(Protocol):
    def __call__(self, node: FlockNode, *args: Any, **kwargs: Any) -> Any:
        ...


class Launcher(ABC):
    """
    Base class for launching functions in an FL process.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def submit(
        self, fn: LauncherFunction, node: FlockNode, /, *args, **kwargs
    ) -> Future:
        raise NotImplementedError()

    @abstractmethod
    def collect(self):
        raise NotImplementedError()
