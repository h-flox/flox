from concurrent.futures import Future
from typing import Any, NewType
from typing import Callable

from flox.flock import FlockNode
from flox.runtime.launcher import Launcher
from flox.runtime.transfer import BaseTransfer

Config = NewType("Config", dict[str, Any])


class Borg:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class Runtime(Borg):
    launcher: Launcher | None = None
    transfer: BaseTransfer | None = None

    def __init__(self, launcher: Launcher, transfer: BaseTransfer):
        Borg.__init__(self)
        self.launcher = launcher
        self.transfer = transfer

    # TODO: Come up with typing for `Job = NewType("Job", Callable[[...], ...])`
    def submit(self, fn: Callable, node: FlockNode, /, *args, **kwargs) -> Future:
        return self.launcher.submit(fn, node, *args, **kwargs, transfer=self.transfer)

    def proxy(self, data: Any):
        return self.transfer.proxy(data)

    # @classmethod
    # def create(
    #     cls, launcher_cfg: Config | None, transfer_cfg: Config | None
    # ) -> "Runtime":
    #     launcher_cfg = {} if launcher_cfg is None else launcher_cfg
    #     transfer_cfg = {} if transfer_cfg is None else transfer_cfg
    #     launcher = Launcher.create(**launcher_cfg)
    #     transfer = Transfer.create(**transfer_cfg)
    #     return Runtime(launcher, transfer)
