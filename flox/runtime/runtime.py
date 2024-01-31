from concurrent.futures import Future
from typing import Callable

from flox.runtime.launcher import Launcher
from flox.runtime.transfer import BaseTransfer
from flox.flock import FlockNode

from typing import Any, NewType

Config = NewType("Config", dict[str, Any])


class Runtime:
    instance: "Runtime"

    def __new__(cls, launcher, transfer):
        if not hasattr(cls, "instance"):
            cls.instance = super(Runtime, cls).__new__(cls, launcher, transfer)
        return cls.instance

    def __init__(self, launcher: Launcher, transfer: BaseTransfer):
        self.launcher = launcher
        self.transfer = transfer

    # TODO: Come up with typing for `Job = NewType("Job", Callable[[...], ...])`
    def submit(self, fn: Callable, node: FlockNode, /, *args, **kwargs) -> Future:
        return self.launcher.submit(fn, node, *args, **kwargs, transfer=self.transfer)

    # @classmethod
    # def create(
    #     cls, launcher_cfg: Config | None, transfer_cfg: Config | None
    # ) -> "Runtime":
    #     launcher_cfg = {} if launcher_cfg is None else launcher_cfg
    #     transfer_cfg = {} if transfer_cfg is None else transfer_cfg
    #     launcher = Launcher.create(**launcher_cfg)
    #     transfer = Transfer.create(**transfer_cfg)
    #     return Runtime(launcher, transfer)
