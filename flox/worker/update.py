from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from flox.flock import FlockNode
from flox.worker.models.base import Params


class FLockNodeStatus:
    pass


class LocalUpdate:
    node: FlockNode
    status: FLockNodeStatus
    params: Params
    extra: dict[str, Any] = None
    sent_timestamp: datetime = field(init=False, default_factory=datetime.now)
    recv_timestamp: datetime = field(init=False, default=None)

    def received(self) -> None:
        self.recv_timestamp = datetime.now()
