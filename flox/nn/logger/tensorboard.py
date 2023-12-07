from datetime import datetime
from .base import BaseLogger
from torch.utils.tensorboard import SummaryWriter
from typing import Any

from flox.flock import FlockNode


class TensorboardLogger(BaseLogger):
    def __init__(self, node: FlockNode, metadata: dict | None = None):
        super().__init__(node, metadata)
        self.writer = SummaryWriter(log_dir=f"runs/{node.path}")

    def log(
        self,
        name: str,
        value: Any,
        timestamp: datetime | None = None,
        round: int | None = None,
        epoch: int | None = None,
        batch_idx: int | None = None,
    ) -> None:
        super().log(name, value, timestamp, round, epoch, batch_idx)
        self.writer.add_scalar(
            name,
            value,
            walltime=timestamp.timestamp(),
            global_step=round, # TODO: handle both round and epoch?
        )

    def clear(self) -> None:
        # TODO: clear out tensorboard directory too?
        return super().clear()
