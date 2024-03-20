from concurrent.futures import Future

from flox.jobs import Job
from flox.runtime.launcher.base import Launcher


class ParslLauncher(Launcher):
    """
    Class that launches tasks via Parsl.
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError(f"{self.__name__} yet implemented")

    def submit(self, job: Job, /, **kwargs) -> Future:
        raise NotImplementedError

    def collect(self):
        raise NotImplementedError
