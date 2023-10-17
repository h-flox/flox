from abc import ABC, abstractmethod
from concurrent.futures import Future

from flox.flock import FlockNode


class FloxExecutor(ABC):
    """
    Base class for executing functions in an FL process.
    """

    def __init__(self):
        pass

    @abstractmethod
    def submit(self, fn, node: FlockNode, /, *args, **kwargs) -> Future:
        raise NotImplementedError()

    @abstractmethod
    def collect(self):
        raise NotImplementedError()
