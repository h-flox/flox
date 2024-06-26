from __future__ import annotations

import typing as t
from concurrent.futures import Future

from flox.process.jobs import Job
from flox.runtime.launcher import Launcher
from flox.runtime.transfer import TransferProtocol

Config = t.NewType("Config", dict[str, t.Any])

if t.TYPE_CHECKING:
    from flox.runtime import Result


class _Borg:
    _shared_state: dict[str, t.Any] = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class Runtime(_Borg):
    launcher: Launcher
    transfer_protocol: TransferProtocol

    def __init__(self, launcher: Launcher, transfer_protocol: TransferProtocol):
        _Borg.__init__(self)
        self.launcher = launcher
        self.transfer_protocol = transfer_protocol

    def submit(self, job: Job, /, **kwargs) -> Future[Result]:
        return self.launcher.submit(job, **kwargs, transfer=self.transfer_protocol)

    def transfer(self, data: t.Any):
        return self.transfer_protocol.transfer(data)
