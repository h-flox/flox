from flox.logger.base import Logger
from flox.topos import Node

from pathlib import Path
from typing import Any
from datetime import datetime


class NullLogger:
    def __init__(self, node: Node | None = None, filename: str | Path | None = None) -> None:
        self.records = []

    def log(
        self,
        name: str,
        value: Any,
        nodeid: str | None = None,
        epoch: int | None = None,
        time: datetime | None = None,
    ) -> None:
        pass

    def log_dict(self, record: dict) -> None:
        pass

    def clear(self) -> None:
        pass
