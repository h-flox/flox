from flox.learn.logger.base import BaseLogger

from typing import Any


class CSVLogger(BaseLogger):
    def __init__(self):
        super().__init__()
        self.records = []

    def log(self, name: str, value: Any) -> None:
        self.records.append({name: value})

    def log_dict(self, record: dict[str, Any]):
        self.records.append(record)
