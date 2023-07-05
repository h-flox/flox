from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class FloxState:
    module: Any
    workers: WorkerConfig
    metrics: dict[str, Any] = field(default=None)
    checkpoints: dict[datetime, Any] = field(default=None)
