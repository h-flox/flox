from __future__ import annotations

import abc
import typing as t
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterable

from flox.learn import FloxModule

if t.TYPE_CHECKING:
    from uuid import UUID


NodeID: t.TypeAlias = int | str
"""..."""


@dataclass(frozen=True)
class Node:
    """
    A node in a Flock.

    Args:
        idx (NodeID): The index of the node within the Topology as a whole (this is assigned by its `Topology`).
        kind (NodeKind): The kind of node.
        globus_compute_endpoint (t.Optional[UUID]): Required if you want to run fitting on Globus Compute;
            defaults to None.
        proxystore_endpoint (t.Optional[UUID]): Required if you want to run fitting with Proxystore
            (recommended if you are using Globus Compute); defaults to None.
    """

    idx: NodeID
    """Assigned during the Flock construction (i.e., not in .yaml/.json file)"""

    kind: NodeKind
    """Which kind of node."""

    globus_compute_endpoint: t.Optional[UUID] = field(default=None)
    """The `globus-compute-endpoint` uuid for using Globus Compute"""

    proxystore_endpoint: t.Optional[UUID] = field(default=None)
    """The `transfer-endpoint` uuid for using Globus Compute"""


class NodeKind(Enum):
    """
    The different kinds of nodes that can exist in a Flock topology.
    """

    COORDINATOR = auto()  # root
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
            "leader": NodeKind.COORDINATOR,
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
            NodeKind.COORDINATOR: "leader",
            NodeKind.AGGREGATOR: "aggregator",
            NodeKind.WORKER: "worker",
        }
        return matches[self]


class NodeState(abc.ABC):
    idx: NodeID
    """The ID of the ``FlockNode`` that the ``NodeState`` corresponds with."""

    cache: dict[str, Any] = field(init=False, default_factory=dict)
    """A dictionary containing extra data. This can be used as a temporary 
    "store" to pass data between callbacks for custom ``Strategy`` objects."""

    def __init__(self, idx: NodeID):
        self.idx = idx
        self.cache = dict()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(idx={self.idx})"

    def __iter__(self) -> Iterable[str]:
        """Returns an iterator through the state's cache."""
        return iter(self.cache)

    def __contains__(self, item) -> bool:
        """Returns `True` if `item` is stored in the state's cache, False otherwise."""
        return item in self.cache

    def __setitem__(self, key: str, value: Any) -> None:
        """Stores a piece of data (``value``) in ``self.extra_data`` using ``key``.

        Parameters:
            key (str): Key to store data in ``self.extra_data``.
            value (Any): Data to store in ``self.extra_data``.

        Examples:
            >>> state = WorkerState(...)
            >>> state["foo"] = "bar"
        """
        self.cache[key] = value

    def __getitem__(self, key: str) -> Any:
        """Retrieves the data in ``self.extra_data`` using ``key``.

        Parameters:
            key (str): Key to retrieve stored data in ``self.extra_data``.

        Examples:
            >>> state = WorkerState(...)
            >>> state["foo"] = "bar"  # Stores the data (see `__setitem__()`).
            >>> print(state["foo"])   # Gets the item.
            >>> # "foo"

        Throws:
            KeyError - Cannot retrieve data at ``key`` if it has not been set.
        """
        return self.cache[key]


class AggrState(NodeState):
    """State of an Aggregator node in a ``Flock``."""

    children: t.Iterable[Node]
    """..."""
    global_model: t.Optional[FloxModule]
    """..."""

    def __init__(
        self,
        idx: NodeID,
        children: t.Iterable[Node],
        global_model: t.Optional[FloxModule],
    ) -> None:
        super().__init__(idx)
        self.children = children
        self.global_model = global_model


class WorkerState(NodeState):
    """State of a Worker node in a ``Flock``."""

    global_model: t.Optional[FloxModule] = None
    """Global model."""
    local_model: t.Optional[FloxModule] = None
    """Local model after local fitting/training."""

    def __init__(
        self,
        idx: NodeID,
        global_model: t.Optional[FloxModule] = None,
        local_model: t.Optional[FloxModule] = None,
    ):
        super().__init__(idx)
        self.global_model = global_model
        self.local_model = local_model
