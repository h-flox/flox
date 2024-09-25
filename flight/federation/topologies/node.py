import typing as t
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID

import pydantic as pyd

from flight.learning import AbstractDataModule, AbstractModule

NodeID: t.TypeAlias = t.Union[int, str]
"""
ID of nodes in Flight topologies; can either be of type `int` or `str`.
"""


class NodeKind(str, Enum):
    """
    Kinds of nodes that can exist in a Flight topology:
    Coordinator (COORD), Aggregator (AGGR), and Worker (WORKER) nodes.
    """

    COORD = "coordinator"
    AGGR = "aggregator"
    WORKER = "worker"


class Node(pyd.BaseModel):
    """A `Node` in Flight.

    An individual `Node` characterizes an endpoint that either takes on the task of
    aggregating model parameters or performing local training. Their connections are
    established with the [`Topology`][flight.federation.topologies.topo.Topology] class.
    """

    idx: NodeID
    """
    The ID of the node.
    """

    kind: NodeKind
    """
    The kind of Node---indicates its *role* in a federation.
    """

    globus_comp_id: UUID | None = pyd.Field(default=None)
    """
    Globus Compute UUID for remote execution.
    """

    proxystore_id: UUID | None = pyd.Field(default=None)
    """
    ProxyStore UUID for data transfer for remote execution with Globus Compute.
    """

    extra: dict[str, t.Any] | None = pyd.Field(default=None)
    """
    Any extra parameters users wish to give to Nodes (e.g., parameters or settings
    around system resource use).
    """

    def __getitem__(self, key: str) -> t.Any:
        """

        Args:
            key:

        Raises:
            - `KeyError`: ...

        Returns:

        """
        if self.extra is None:
            raise KeyError("This Node does not have an `extra` cache.")
        return self.extra[key]

    def __setitem__(self, key: str, value: t.Any) -> None:
        """
        Setter method for storing data into the Node's `extra` cache.

        Args:
            key (str): Key to store datum.
            value (typing.Any): Datum to store into node's `extra` cache.
        """
        if self.extra is None:
            self.extra = {}
        self.extra[key] = value

    def __hash__(self):
        # TODO: Re-investigate.
        return hash(self.idx)

    def get(self, key: str, backup_value: t.Any) -> t.Any:
        if self.extra is None:
            raise KeyError(
                "This node's `extra` cache is `None`. "
                "You should initialize this directly via `my_node.extra = {}`."
            )
        return self.extra.get(key, backup_value)


@dataclass
class NodeState:
    """
    Dataclass that wraps the state of a node during a federation.

    Args:
        idx (NodeID): The ID of the node.

    Throws:
        - TypeError: This class cannot be directly instantiated. Only its children
          classes can be instantiated.
    """

    idx: NodeID
    cache: dict[str, t.Any] = field(
        init=False, default_factory=dict, repr=False, hash=False
    )

    def __post_init__(self):
        if type(self) is NodeState:
            raise TypeError(
                "Cannot instantiate an instance of `NodeState`. Instead, you must "
                "instantiate instances of `WorkerState` or `AggrState`."
            )

    def __getitem__(self, key: str) -> t.Any:
        """
        Getter method that fetches a data item from the state's cache by key.

        Args:
            key (str): Name of item to retrieve from `cache`.

        Raises:
            - KeyError: Thrown if a key that is not in the `cache`.

        Examples:
            >>> state = NodeState(0)
            >>> state["foo"] = "bar"
            >>> state["foo"]
            'bar'

        Returns:
            The cached datum.
        """
        return self.cache[key]

    def __setitem__(self, key: str, value: t.Any) -> None:
        """
        Setter function that stores a data item into the state's cache by key.

        Args:
            key (str): The key to store the data in cache for lookup.
            value (typing.Any): The data to store in the cache.
        """
        self.cache[key] = value

    def get(self, key: str, backup_value: t.Any = None) -> t.Any:
        """
        Implements the same functionality as the standard `dict.get()` method.

        Fetches any data stored at in the cache with the provided key. However, if not
        such data exists, then the default `backup_value` will be returned instead.
        If no backup value is given, then `None` is returned.

        Args:
            key (str): Name of the item to retrieve from `cache`.
            backup_value (typing.Any): The value returned if `key` is not
                found in `cache`.

        Examples:
            >>> state = NodeState(0)
            >>> state.get("foo")
            None
            >>> state.get("foo", "bar")
            'bar'

        Returns:
            The cached datum or the provided backup value.
        """
        return self.cache.get(key, backup_value)


@dataclass
class AggrState(NodeState):
    """
    The state of an Aggregator node.

    Args:
        children (t.Iterable[Node]): Child nodes in the topology.
        aggr_model (AbstractModule | None): Aggregated model.
    """

    children: t.Sequence[Node]
    aggr_model: AbstractModule | None = None


@dataclass
class WorkerState(NodeState):
    """
    The state of a Worker node.

    Args:
        global_model (AbstractDataModule | None): ...
        local_model (AbstractDataModule | None): ...
    """

    global_model: AbstractDataModule | None = None
    local_model: AbstractDataModule | None = None
