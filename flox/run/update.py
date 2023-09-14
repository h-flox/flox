from dataclasses import dataclass
from pandas import DataFrame
from typing import Any

from flox.flock import FlockNodeKind, FlockNodeID
from flox.flock.states import NodeState
from flox.typing import StateDict


@dataclass
class TaskUpdate:
    """A simple dataclass that is returned by tasks executed on Aggregator and Worker nodes in a ``Flock``.

    Aggregators and Worker nodes have to return the same type of object to support hierarchical execution.
    """

    node_state: NodeState
    """The state of the ``Flock`` node based on its kind."""

    node_idx: FlockNodeID
    """The ID of the ``Flock`` node."""

    node_kind: FlockNodeKind
    """The kind of the ``Flock`` node."""

    state_dict: StateDict
    """The ``StateDict`` of the PyTorch module (either aggregated or trained locally)."""

    history: DataFrame
    """The history of results."""
