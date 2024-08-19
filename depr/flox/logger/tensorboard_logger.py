from __future__ import annotations

import typing as t
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

if t.TYPE_CHECKING:
    from flight.federation.topologies import Node, NodeID


class TensorBoardLogger:
    def __init__(
        self,
        node: Node | None = None,
        filename: str | Path | None = None,
    ) -> None:
        self.records = []

        if node:
            self.writer = SummaryWriter(log_dir=f"./runs/{node.idx}")
        else:
            self.writer = SummaryWriter(log_dir="./runs")

    def log(
        self,
        name: str,
        value: t.Any,
        node_idx: NodeID | None = None,
        epoch: int | None = None,
        time: datetime | None = None,
    ) -> None:
        self.records.append(
            {
                "name": name,
                "value": value,
                "node_idx": node_idx,
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
