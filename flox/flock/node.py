from dataclasses import dataclass
from enum import Enum, auto
from typing import NewType, Optional, Sequence
from uuid import UUID

FlockNodeID = NewType("FlockNodeID", int)


class FlockNodeKind(Enum):
    LEADER = auto()  # root
    AGGREGATOR = auto()  # middle
    WORKER = auto()  # leaf

    @staticmethod
    def from_str(s: str) -> "FlockNodeKind":
        s = s.lower().strip()
        matches = {
            "leader": FlockNodeKind.LEADER,
            "aggregator": FlockNodeKind.AGGREGATOR,
            "worker": FlockNodeKind.WORKER,
        }
        if s in matches:
            return matches[s]
        else:
            raise ValueError(
                f"Illegal `str` value given to `FlockNodeKind.from_str()`. "
                f"Must be one of the following: {list(matches.keys())}."
            )

    def to_str(self) -> str:
        matches = {
            FlockNodeKind.LEADER: "leader",
            FlockNodeKind.AGGREGATOR: "aggregator",
            FlockNodeKind.WORKER: "worker",
        }
        return matches[self]


@dataclass(frozen=True)
class FlockNode:
    idx: FlockNodeID  # Assigned during the Flock construction (i.e., not in .yaml/.json file)
    kind: FlockNodeKind
    globus_compute_endpoint: Optional[UUID]
    proxystore_endpoint: Optional[UUID]
    # children_idx: Optional[Sequence[FlockNodeID]]
