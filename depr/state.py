from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from depr.flock import Flock


@dataclass
class FloxState:
    module: Any
    flock: Flock
    metrics: dict[str, Any] = field(default=None)
    checkpoints: dict[datetime, Any] = field(default=None)
