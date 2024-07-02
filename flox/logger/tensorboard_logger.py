from datetime import datetime
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from flox.federation.topologies import Node


class TensorBoardLogger:
    def __init__(
        self, node: Node | None = None, filename: str | Path | None = None
    ) -> None:
        self.records = []

        if node:
            self.writer = SummaryWriter(log_dir=f"./runs/{node.idx}")
        else:
            self.writer = SummaryWriter(log_dir="./runs")

    def log(
        self,
        name: str,
        value: Any,
        nodeid: str | None = None,
        epoch: int | None = None,
        time: datetime | None = None,
    ) -> None:
        self.records.append(
            {
                "name": name,
                "value": value,
                "nodeid": nodeid,
                "epoch": epoch,
                "datetime": time,
            }
        )

        self.writer.add_scalar(
            name, value, global_step=epoch, walltime=time.timestamp()
        )

    def log_dict(self, record: dict) -> None:
        self.records.append(record)
        self.writer.add_scalar(
            record["name"],
            record["value"],
            global_step=record["epoch"],
            walltime=record["datetime"].timestamp(),
        )

    def clear(self) -> None:
        self.records = []
        self.writer.close()
