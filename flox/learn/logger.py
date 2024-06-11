from typing import Any

from pandas import DataFrame


class ModelLogger:
    """Class that logs metrics for results."""

    def __init__(self):
        self.records = []

    def log(self, name: str, value: Any) -> None:
        self.records.append({name: value})

    def log_dict(self, record: dict[str, Any]):
        self.records.append(record)

    def clear(self):
        self.records = []

    def dataframe(self) -> DataFrame:
        return DataFrame.from_records(self.records)
