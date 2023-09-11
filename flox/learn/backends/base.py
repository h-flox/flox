from abc import ABC
from concurrent.futures import Future

from flox.flock import FlockNode


class FloxExecutor(ABC):
    """
    Base class for executing functions in an FL process.
    """

    def __init__(self):
        pass

    def submit(self, fn, node: FlockNode, /, *args, **kwargs) -> Future:
        raise NotImplementedError()

    def collect(self):
        pass
