from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from flox.federation.topologies import Node, NodeID

Record: t.TypeAlias = t.Dict[str, t.Any]


@t.runtime_checkable
class Logger(t.Protocol):
    records: list

    def __init__(
        self, node: Node | None = None, filename: str | Path | None = None
    ) -> None:
        self.records = []

    def log(
        self,
        name: str,
        value: t.Any,
        node_id: NodeID,
        epoch: int | None,
        time: datetime | None,
    ) -> None:
        """
        log each value passed in with its correct key

        args:
            name (Str): type to be logged
            value (Any): the value of the type to be logged
            nodeid (str | None): identifies which node the log pertains to
            epoch (int | None): training round
            time (datetime | None): time of log
        """

    def log_dict(self, record: dict) -> None:
        """
        log/add a dictionary to records
        Args:
            record (dict): set of keys and values to be logged
        """

    def clear(self) -> None:
        """
        clears the records list
        """
