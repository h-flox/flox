from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

if t.TYPE_CHECKING:
    from flight.learning.module import Params
    from flight.state import AbstractNodeState
    from flight.system.topology import Node


@dataclass
class Result:
    node: Node
    state: AbstractNodeState
    params: Params
    extra: dict[str, t.Any] = field(default_factory=dict)
