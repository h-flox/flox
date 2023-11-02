from concurrent.futures import Future
from flox.flock import FlockNode
from flox.backends.launcher.base import Launcher


class ParslLauncher(Launcher):
    """
    Class that launched tasks via Parsl.
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError(f"{self.__name__} yet implemented")

    def submit(self, fn, node: FlockNode, /, *args, **kwargs) -> Future:
        pass

    def collect(self):
        pass
