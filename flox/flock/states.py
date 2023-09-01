from dataclasses import dataclass, field
from typing import Any, NewType, Optional, Union

import torch


@dataclass
class FloxAggregatorState:
    pass


@dataclass(repr=False)
class FloxWorkerState:
    pre_local_train_model: torch.nn.Module  # Global model.
    post_local_train_model: Optional[torch.nn.Module] = None
    extra_data: dict[str, Any] = field(default_factory=dict)

    def __setitem__(self, key: str, value: Any) -> None:
        self.extra_data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.extra_data[key]


NodeState = NewType("NodeState", Union[FloxAggregatorState, FloxWorkerState])
