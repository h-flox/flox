import torch

from dataclasses import dataclass, field
from typing import Any


@dataclass(repr=False)
class FloxWorkerState:
    current_model: torch.nn.Module
    global_model: torch.nn.Module
    extra_data: dict[str, Any] = field(default=None)
