import enum
import typing as t
from dataclasses import dataclass, field
from uuid import UUID

from .types import NodeID


class NodeKind(enum.Enum):
    """Enum for the kind of nodes that can exist in a Flight topology."""

    COORDINATOR = "coordinator"
    """Coordinator node. There can only be 1 in a topology."""

    AGGREGATOR = "aggregator"
    """Aggregator nodes."""

    WORKER = "worker"
    """Worker nodes."""


class Node:
    idx: NodeID
    """The ID of the node."""

    kind: NodeKind

    globus_compute_id: UUID | None
    """Globus Compute UUID for remote execution."""

    proxystore_id: UUID | None
    """ProxyStore UUID for data transfer across remote endpoints with Globus Compute."""

    extra: dict[str, t.Any]
    """Extra parameters users with to give to Nodes that can be used during federations."""

    def __init__(
        self,
        idx: NodeID,
        kind: NodeKind | str,
        globus_compute_id: UUID | None = None,
        proxystore_id: UUID | None = None,
        **kwargs,
    ) -> None:
        self.idx = idx
        self.kind = kind if isinstance(kind, NodeKind) else NodeKind(kind)
        self.globus_compute_id = globus_compute_id
        self.proxystore_id = proxystore_id
        self.extra = {}

        for key, value in kwargs.items():
            self.extra[key] = value

    def __repr__(self) -> str:
        template = "{}(id={}{})"
        kind = self.kind.value.title()

        if all(
            [
                self.globus_compute_id is None,
                self.proxystore_id is None,
                len(self.extra) == 0,
            ]
        ):
            return template.format(kind, self.idx, "")

        other = ""
        if self.globus_compute_id is not None:
            other += f", globus-compute-id={self.globus_compute_id}"
        if self.proxystore_id is not None:
            other += f", proxystore-id={self.proxystore_id}"
        if len(self.extra):
            other += ", extra=["
            other += ", ".join(key for key in list(self.extra))
            other += "]"

        return template.format(kind, self.idx, other)

    def __getitem__(self, key: str) -> t.Any:
        """
        Getter method that grabs data by the provided `key` from the
        `extra` attribute.

        Args:
            key (str): Key to fetch from `self.extra`.

        Returns:
            The datum stored at `self.extra[key]`.

        Throws:
            - `KeyError`: If `key` is not found in `self.extra`.
        """
        return self.extra[key]

    def __setitem__(self, key: str, value: t.Any) -> None:
        """
        Setter method for storing data into the `extra` attribute.

        Args:
            key (str): Key to store value.
            value (t.Any): Datum to store into the node's `extra` attribute.
        """
        self.extra[key] = value

    def get(self, key: str, default: t.Any = None) -> t.Any:
        """
        Getter method with support for default value.

        Args:
            key (str): Key to get value from.
            default (t.Any): Default value to return if no value exists at `key`.
                Defaults to `None`.

        Returns:
            Value stored at `key` or the default value.
        """
        return self.extra.get(key, default)
