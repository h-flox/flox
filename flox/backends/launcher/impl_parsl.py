from concurrent.futures import Future
from flox.flock import FlockNode
from flox.backends.launcher.impl_base import Launcher


class ParslLauncher(Launcher):
    """
    Class that launches tasks via Parsl.
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError(f"{self.__name__} yet implemented")

    def submit(self, fn, node: FlockNode, /, *args, **kwargs) -> Future:
        pass

    def collect(self):
        pass
