from dataclasses import dataclass
from enum import Enum, auto
from uuid import UUID

NodeID = int | str  # NewType("NodeID", int | str)


class NodeKind(Enum):
    """
    The different kinds of nodes that can exist in a Flock topology.
    """

    LEADER = auto()  # root
    AGGREGATOR = auto()  # middle
    WORKER = auto()  # leaf

    @staticmethod
    def from_str(s: str) -> "NodeKind":
        """
        Converts a string (namely, 'leader', 'aggregator', and 'worker') into their respective item in this Enum.

        For convenience, this function is *not* sensitive to capitalization or trailing whitespace (i.e.,
        `NodeKind.from_str('LeaAder  ')` and `NodeKind.from_str('leader')` are both valid and equivalent).

        Args:
            s (str): String to convert into the respective Enum item.

        Throws:
            ValueError: Thrown by illegal string values do not match the above description.

        Returns:
            NodeKind corresponding to the passed in String.
        """
        s = s.lower().strip()
        matches = {
            "leader": NodeKind.LEADER,
            "aggregator": NodeKind.AGGREGATOR,
            "worker": NodeKind.WORKER,
        }
        if s in matches:
            return matches[s]
        raise ValueError(
            f"Illegal `str` value given to `NodeKind.from_str()`. "
            f"Must be one of the following: {list(matches.keys())}."
        )

    def to_str(self) -> str:
        """
        Returns the string representation of the Enum item.

        Returns:
            String corresponding to the NodeKind.
        """
        matches = {
            NodeKind.LEADER: "leader",
            NodeKind.AGGREGATOR: "aggregator",
            NodeKind.WORKER: "worker",
        }
        return matches[self]


@dataclass(frozen=True)
class FlockNode:
    """
    A node in a Flock.

    Args:
        idx (NodeID): The index of the node within the Flock as a whole (this is assigned by its `Flock`).
        kind (NodeKind): The kind of node.
        globus_compute_endpoint (UUID | None): Required if you want to run fitting on Globus Compute;
            defaults to None.
        proxystore_endpoint (UUID | None): Required if you want to run fitting with Proxystore
            (recommended if you are using Globus Compute); defaults to None.
    """

    idx: NodeID
    """Assigned during the Flock construction (i.e., not in .yaml/.json file)"""

    kind: NodeKind
    """Which kind of node."""

    globus_compute_endpoint: UUID | None = None
    """The `globus-compute-endpoint` uuid for using Globus Compute"""

    proxystore_endpoint: UUID | None = None
    """The `transfer-endpoint` uuid for using Globus Compute"""
