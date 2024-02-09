from collections.abc import Iterable
from dataclasses import field
from typing import Any

import torch

from flox.flock import FlockNodeID


class NodeState:
    idx: FlockNodeID
    """The ID of the ``FlockNode`` that the ``NodeState`` corresponds with."""

    cache: dict[str, Any] = field(default_factory=dict)
    """A dictionary containing extra data. This can be used as a temporary "store" to pass data between
    callbacks for custom ``Strategy`` objects."""

    def __init__(self, idx: FlockNodeID):
        if type(self) is NodeState:
            raise NotImplementedError(
                "Cannot instantiate instance of ``NodeState`` (must instantiate instance of "
                "subclasses: ``FloxAggregatorState`` or ``FloxWorkerState``)."
            )
        self.idx = idx
        self.cache = {}

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
            >>> state = FloxWorkerState(...)
            >>> state["foo"] = "bar"
        """
        self.cache[key] = value

    def __getitem__(self, key: str) -> Any:
        """Retrieves the data in ``self.extra_data`` using ``key``.

        Parameters:
            key (str): Key to retrieve stored data in ``self.extra_data``.

        Examples:
            >>> state = FloxWorkerState(...)
            >>> state["foo"] = "bar"  # Stores the data (see `__setitem__()`).
            >>> print(state["foo"])   # Gets the item.
            >>> # "foo"

        Throws:
            KeyError - Cannot retrieve data at ``key`` if it has not been set.
        """
        return self.cache[key]


class FloxAggregatorState(NodeState):
    """State of an Aggregator node in a ``Flock``."""

    def __init__(self, idx: FlockNodeID):
        super().__init__(idx)


class FloxWorkerState(NodeState):
    """State of a Worker node in a ``Flock``."""

    pre_local_train_model: torch.nn.Module | None = None
    """Global model."""

    post_local_train_model: torch.nn.Module | None = None
    """Local model after local fitting/training."""

    def __init__(
        self,
        idx: FlockNodeID,
        pre_local_train_model: torch.nn.Module | None = None,
        post_local_train_model: torch.nn.Module | None = None,
    ):
        super().__init__(idx)
        self.pre_local_train_model = pre_local_train_model
        self.post_local_train_model = post_local_train_model

    def __repr__(self) -> str:
        template = (
            "FloxWorkerState(pre_local_train_model={}, post_local_train_model={})"
        )
        return template.format(self.pre_local_train_model, self.post_local_train_model)


# NodeState = NewType("NodeState", Union[FloxAggregatorState, FloxWorkerState])
# """A `Type` included for convenience. It is equivalent to ``Union[FloxAggregatorState, FloxWorkerState]``."""
