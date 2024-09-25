from __future__ import annotations

import typing as t
from datetime import datetime
from pathlib import Path
from typing import Any

from flox.federation.topologies import Node

if t.TYPE_CHECKING:
    from flox.logger.base import Record


class NullLogger:
    def __init__(
        self, node: Node | None = None, filename: str | Path | None = None
    ) -> None:
        self.records: t.List[Record] = []

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
