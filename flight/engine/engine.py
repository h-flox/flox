from __future__ import annotations

import typing as t
from concurrent.futures import Future

from flight.engine.control.serial import SerialController

if t.TYPE_CHECKING:
    from .control.base import AbstractController
    from .data.base import AbstractTransfer, BaseTransfer


class Engine:
    control_plane: AbstractController
    data_plane: AbstractTransfer

    def __init__(self):
        self.control_plane = SerialController()
        self.data_plane = BaseTransfer()

    def __call__(self, fn, *args, **kwargs) -> Future:
        return self.control_plane(fn, *args, **kwargs)

    def transfer(self, data: t.Any) -> t.Any:
        return self.data_plane(data)
