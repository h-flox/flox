from typing import Any

import pandas as pd


class BaseLogger:
    def __init__(self):
        self.records = []

    def log(self, name: str, value: Any):
        ...

    def log_dict(self, record):
        for name, value in record.items():
            self.log(name, value)

    def clear(self) -> None:
        self.records = []

    def to_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame.from_records(self.records)
        for col in df.columns:
            if "time" in col:
                df[col] = pd.to_datetime(df[col])
        return df
