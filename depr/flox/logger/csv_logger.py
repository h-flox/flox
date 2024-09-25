import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from flox.federation.topologies import Node


class CSVLogger:
    def __init__(
        self, node: Node | None = None, filename: str | Path | None = None
    ) -> None:
        self.records = []
        self.filename = filename

    def log(
        self,
        name: str,
        value: Any,
        nodeid: str | None = None,
        epoch: int | None = None,
        time: datetime | None = None,
    ) -> None:
        if not self.records and self.filename:
            f = open(self.filename, "w")
            f.truncate()
            f.close()
        data = {
            "name": name,
            "value": value,
            "nodeid": nodeid,
            "epoch": epoch,
            "datetime": time,
        }

        self.records.append(data)
        if self.filename:
            with open(self.filename, "a") as outfile:
                writer = csv.writer(outfile)
                writer.writerow(data.values())

    def log_dict(self, record: dict) -> None:
        if not self.records and self.filename:
            f = open(self.filename, "w")
            f.truncate()
            f.close()
        self.records.append(record)

        if self.filename:
            with open(self.filename, "a") as outfile:
                writer = csv.writer(outfile)
                writer.writerow(record.values())

    def to_pandas(self, filename: str | Path | None = None) -> None | str:
        df = pd.DataFrame.from_records(self.records)
        return df.to_csv(filename, index=False)

    def clear(self) -> None:
        self.records = []
