from abc import ABC, abstractmethod
from concurrent.futures import Future

from flox.federation.jobs import Job

# @dataclass
# class LauncherConfig:
#     kind: LauncherKind
#     args: LauncherArgs


class Launcher(ABC):
    """
    Base class for launching functions in an FL federation.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def submit(self, job: Job, /, **kwargs) -> Future:
        raise NotImplementedError()

    @abstractmethod
    def collect(self):
        # TODO: Check if this is needed at all.
        raise NotImplementedError()
