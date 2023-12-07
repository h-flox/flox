from typing import Any
from datetime import datetime

from flox.flock import FlockNode

import pandas as pd


class BaseLogger:
    def __init__(self, node: FlockNode, metadata: dict | None = None):
        self.node = node
        self.metadata = metadata or {}
        self.records = []

    def log(
        self,
        name: str,
        value: Any,
        timestamp: datetime | None = None,
        round: int | None = None,
        epoch: int | None = None,
        batch_idx: int | None = None,
    ) -> None:
        self.records.append({
            "timestamp": timestamp or datetime.now(),
            "round": round,
            "epoch": epoch,
            "batch_idx": batch_idx,
            **self.metadata, # metadata overrides default logs
            name: value, # name/value overrides everything
        })

    def clear(self) -> None:
        self.records = []

    def to_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame.from_records(self.records)
        for col in df:
            if "time" in col:
                df[col] = pd.to_datetime(df[col])
        return df
