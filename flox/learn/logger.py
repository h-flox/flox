import typing as t

from pandas import DataFrame


class ModelLogger:
    """Class that logs metrics for results."""

    def __init__(self):
        self.records = []

    def log(self, name: str, value: t.Any) -> None:
        self.records.append({name: value})

    def log_dict(self, record: dict[str, t.Any]):
        self.records.append(record)

    def clear(self):
        self.records = []

    def dataframe(self) -> DataFrame:
        return DataFrame.from_records(self.records)


class NewLogger:
    def __enter__(self):
        print("Entering `NewLogger` context.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting `NewLogger` context")
