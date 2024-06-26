import typing as t
from concurrent.futures import Future

from flox.process.jobs import Job
from flox.runtime.launcher import Launcher
from flox.runtime.transfer import TransferProtocol
from flox.runtime.result import Result

Config = t.NewType("Config", dict[str, t.Any])

if t.TYPE_CHECKING:
    from flox.runtime import Result


class _Borg:
    _shared_state: dict[str, t.Any] = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class Runtime(_Borg):
    launcher: Launcher
    transfer: TransferProtocol

    def __init__(self, launcher: Launcher, transfer: TransferProtocol):
        _Borg.__init__(self)
        self.launcher = launcher
        self.transfer = transfer

    def submit(self, job: Job, /, **kwargs) -> Future[Result]:
        return self.launcher.submit(job, **kwargs, transfer=self.transfer)

    def transfer(self, data: t.Any):
        return self.transfer.transfer(data)
