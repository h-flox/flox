from abc import ABC, abstractmethod
from concurrent.futures import Future

from flox.flock import FlockNode


#
# def Launcher


# @dataclass
# class LauncherConfig:
#     kind: LauncherKind
#     args: LauncherArgs


class Launcher(ABC):
    """
    Base class for launching functions in an FL process.
    """

    def __init__(self):
        pass

    @abstractmethod
    def submit(self, fn, node: FlockNode, /, *args, **kwargs) -> Future:
        raise NotImplementedError()

    @abstractmethod
    def collect(self):
        raise NotImplementedError()
