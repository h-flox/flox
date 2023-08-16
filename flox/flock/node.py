from dataclasses import dataclass
from enum import Enum, auto
from typing import NewType, Optional, Sequence
from uuid import UUID


FlockNodeID = NewType("FlockNodeID", int)


class FlockNodeKind(Enum):
    """
    The differnet types of nodes that can exist in a FLock topology.
    """

    LEADER = auto()  # root
    AGGREGATOR = auto()  # middle
    WORKER = auto()  # leaf

    @staticmethod
    def from_str(s: str) -> "FlockNodeKind":
        """
        Converts a string (namely, 'leader', 'aggregator', and 'worker') into their respective item in this Enum.
        For convenience, this function is *not* sensitive to capitalization or trailing whitespace (i.e.,
        `FlockNodeKind.from_str('LeaAder  ')` and `FlockNodeKind.from_str('leader')` are both valid and equivalent).

        Args:
            s (str): Text to convert into the respective Enum item.

        Throws:
            ValueError: Thrown by illegal string values do not match the above description.

        Returns:
            FlockNodeKind
        """
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
        """
        Returns the string representation of the Enum item.

        Returns:
            str
        """
        matches = {
            FlockNodeKind.LEADER: "leader",
            FlockNodeKind.AGGREGATOR: "aggregator",
            FlockNodeKind.WORKER: "worker",
        }
        return matches[self]


@dataclass(frozen=True)
class FlockNode:
    """
    A node in a Flock.

    Args:
        idx (FlockNodeID): The index of the node within the Flock as a whole (this is assigned by its `Flock`).
        kind (FlockNodeKind): The kind of node.
        globus_compute_endpoint (Optional[UUID]): Required if you want to run fitting on Globus Compute;
            defaults to None.
        proxystore_endpoint (Optional[UUID]): Required if you want to run fitting with Proxystore
            (recommended if you are using Globus Compute); defaults to None.
    """

    idx: FlockNodeID  # Assigned during the Flock construction (i.e., not in .yaml/.json file)
    kind: FlockNodeKind
    globus_compute_endpoint: Optional[UUID] = None
    proxystore_endpoint: Optional[UUID] = None
