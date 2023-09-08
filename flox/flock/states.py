from dataclasses import dataclass, field
from typing import Any, NewType, Optional, Union

import torch


@dataclass
class FloxAggregatorState:
    """State of an Aggregator node in a ``Flock``."""

    pass


@dataclass(repr=False)
class FloxWorkerState:
    """State of a Worker node in a ``Flock``."""

    pre_local_train_model: torch.nn.Module
    """Global model."""

    post_local_train_model: Optional[torch.nn.Module] = None
    """Local model after local fitting/training."""

    extra_data: dict[str, Any] = field(default_factory=dict)
    """A dictionary containing extra data. This can be used as a temporary "store" to pass data between
    callbacks for custom ``Strategy`` objects."""

    def __setitem__(self, key: str, value: Any) -> None:
        """Stores a piece of data (``value``) in ``self.extra_data`` using ``key``.

        Parameters:
            key (str): Key to store data in ``self.extra_data``.
            value (Any): Data to store in ``self.extra_data``.

        Examples:
            >>> state = FloxWorkerState(...)
            >>> state["foo"] = "bar"
        """
        self.extra_data[key] = value

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
        return self.extra_data[key]


NodeState = NewType("NodeState", Union[FloxAggregatorState, FloxWorkerState])
"""A `Type` included for convenience. It is equivalent to ``Union[FloxAggregatorState, FloxWorkerState]``."""
