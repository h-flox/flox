from __future__ import annotations

import typing as t
from concurrent.futures import Future

from flight.engine.control.serial import SerialCP
from flight.engine.data.base import BaseTransfer

if t.TYPE_CHECKING:
    from .control.proto import ControlPlane
    from .data.proto import TransferProto


class Engine:
    control_plane: ControlPlane
    data_plane: TransferProto

    def __init__(self):
        self.control_plane = SerialCP()
        self.data_plane = BaseTransfer()

    def __call__(self, fn, *args, **kwargs) -> Future:
        return self.control_plane(fn, *args, **kwargs)

    def transfer(self, data: t.Any) -> t.Any:
        return self.data_plane(data)
