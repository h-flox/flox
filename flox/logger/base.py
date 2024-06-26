from typing import Protocol, Any, runtime_checkable
from datetime import datetime
from flox.topos import Node
from pathlib import Path


@runtime_checkable
class Logger(Protocol):
    records: list

    def __init__(
        self, node: Node | None = None, filename: str | Path | None = None
    ) -> None:
        self.records = []

    def log(
        self,
        name: str,
        value: Any,
        nodeid: str | None,
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
