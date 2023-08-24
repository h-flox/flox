import torch

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(repr=False)
class FloxWorkerState:
    global_model: torch.nn.Module
    pre_local_train_model: torch.nn.Module
    post_local_train_model: Optional[torch.nn.Module] = None
    extra_data: dict[str, Any] = field(default=None)
