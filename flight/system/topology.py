from __future__ import annotations

import pathlib
import typing as t
from uuid import UUID

import networkx as nx

from .exceptions import NodeNotFoundError, TopologyException
from .io import (
    from_adj_list,
    from_adj_matrix,
    from_dict,
    from_json,
    from_networkx,
    from_yaml,
)
from .node import Node, NodeKind
from .types import Edge, GraphDict, NodeID

if t.TYPE_CHECKING:
    from numpy.typing import ArrayLike


class Topology:
    """
    Defines a topology where the connections between `Node`s are
    established.
    """

    def __init__(
        self,
        nodes: list[Node],
        edges: list[Edge],
        _source: pathlib.Path | str | None = None,
    ):
        """Constructs a topology for federation.

        Args:
            nodes (list[Node]): List of `Node`s in the `Topology`.
            edges (list[Edge]): List of edges between `Node`s that
                exist in the `Topology`.
            _source (pathlib.Path | str | None, optional): Indicates
                the file used to instantiate the `Topology` (if one
                was used). This should *not* be used by the user;
                instead it should be used only by factory methods.
                Defaults to `None`.

        Raises:
            - `ValueError`: Raised if value for `_source` it not a
              `pathlib.Path`, `str`, or `None`.
        """
        node_dict = {node.idx: node for node in nodes}
        graph: nx.DiGraph = nx.DiGraph()
        graph.add_nodes_from(list(node_dict))
        graph.add_edges_from(edges)

        self._nodes = node_dict
        self._edges = edges
        self._graph = graph

        match _source:
            case pathlib.Path() | None:
                self._source = _source
            case str():
                self._source = pathlib.Path(_source)
            case _:
                raise ValueError(
                    "Illegal argument kind for `source`. "
                    "Tip: This should argument *not* be set by the user. "
                    "Leave it to factory methods."
                )

    def __contains__(self, node: Node | NodeID) -> bool:
        """Return `True` if the node (`Node` or `NodeID`) is in the `Topology`,
        `False`, otherwise.

        Args:
            node (Node | NodeID): The `Node` or `NodeID` to check in the `Topology`.

        Returns:
            `True` if `node` exists in the `Topology`; `False` otherwise.

        Throws:
            - `ValueError`: Illegal value given for argument `node`.
        """
        if isinstance(node, Node):
            return node.idx in self._nodes
        elif isinstance(node, NodeID):
            return node in self._nodes
        raise ValueError("Illegal value for argument `node`.")

    def __iter__(self) -> t.Iterator[NodeID]:
        """Returns an iterator through the `NodeID`s of each `Node` in the `Topology`.

        Returns:
            Iterator through every `NodeID` in the `Topology`.
        """
        for node in self.nodes():
            yield node.idx

    def __getitem__(self, idx: NodeID) -> Node:
        """Get the `Node` in the `Topology` by `NodeID`.

        Args:
            idx (NodeID): ID of the `Node` to return.

        Returns:
            The `Node` in the `Topology` with the given `NodeID`.

        Throws:
            - `NodeNotFoundError`: Thrown if no `Node` with the given `NodeID` exists in
              the `Topology`.
        """
        if idx not in self:
            raise NodeNotFoundError()
        return self._nodes[idx]

    def __len__(self) -> int:
        """Total number of nodes in the `Topology`.

        Returns:
            Number of nodes in the `Topology`.
        """
        return len(self.nodes())

    def number_of_edges(self) -> int:
        """
        Returns the number of edges in the `Topology`.

        Returns:
            Number of edges.
        """
        return len(self.edges)

    def number_of_nodes(self, kind: NodeKind | str | None = None) -> int:
        """
        Returns the number of nodes in the `Topology` (of a designated
        kind if specified).

        Args:
            kind (NodeKind | str | None): Specifies the kind of nodes
                to count. This is `None` by default, which will return
                the total number of nodes in the

        Returns:
            The number of nodes of a specified kind (if one is given).
                Topology.

        Notes:
            If you wish to use no argument (i.e., `topo.number_of_nodes()`),
            it is recommended that you instead use `len()`. The `number_of_nodes()`
            method iterates through the nodes, whereas the implementation for
            `len()` does not.
        """
        return sum(1 for _ in self.nodes(kind))

    def nodes(self, kind: NodeKind | str | None = None) -> list[Node]:
        """Creates an iterator that loops through nodes making up the topology.

        The optional parameter, `kind`, can be used to filter only certain nodes to be
        included in the returned iterator.

        Args:
            kind (NodeKind | str | None, optional): The kind of nodes to include in the
                iterator. If `None`, then all nodes in the topology are included in the
                returned iterator. Defaults to `None`.

        Returns:
            A list containing the nodes of the indicated type (if specified).

        Examples:
            >>> nodes: list[Node] = ...
            >>> edges: list[Edge] = ...
            >>> topo = Topology(nodes, edges)
            >>> for node in topo.nodes(kind="coordinator"):
            >>>     print(node)
            Coordinator(id=0)

        Throws:
            - `ValueError`: thrown in the event the user provides an illegal `str` for
                arg `kind` (see docs for `NodeKind` enum).

        """
        if kind is None:
            return list(self._nodes.values())

        if isinstance(kind, str):
            kind = NodeKind(kind)

        nodes = []
        for idx in self:
            if kind is None or self[idx].kind is kind:
                nodes.append(self[idx])

        return nodes

    def parent(self, node: Node | NodeID) -> Node:
        """
        Get the parent of the given `Node` in the `Topology`.

        Args:
            node (Node | NodeID): The node to get the parent of.

        Returns:
            The parent of the given node.

        Throws:
            - `NodeNotFoundError`: If the node is not found in the
              `Topology`.
            - `TopologyException`: If the node is the coordinator,
              which cannot have a parent.
        """
        idx = resolve_node_or_idx(node)
        if idx not in self:
            raise NodeNotFoundError

        node = self[idx]
        if node.kind is NodeKind.COORDINATOR:
            raise TopologyException(f"{node.kind.value.title()} node has no parent.")

        par = next(iter(self.graph.predecessors(idx)))
        return self[par]

    def children(self, node: Node | NodeID) -> list[Node]:
        """Returns a list of the children node of a `Node` in the topology.

        Args:
            node: (Node | NodeID): The node to get the children of.

        Returns:
            List of child nodes.
        """
        idx = resolve_node_or_idx(node)
        return [child for child in self.graph.successors(idx)]

    def globus_compute_ready(self) -> bool:
        """Returns a boolean for whether the given `Topology` has the necessary data
        on each node to use ProxyStore.

        Returns:
            `True` if the given `Topology` has all information to use ProxyStore,
            `False` otherwise.

        Throws:
            - `ValueError`: In the event that a `Node`'s `proxystore_id` attribute
              is a `str` that cannot be converted into a proper `UUID`.
        """
        for node in self.nodes():
            if node.kind is NodeKind.COORDINATOR:
                continue
            globus_id = node.globus_compute_id
            if globus_id is None or not isinstance(globus_id, UUID):
                return False

        return True

    def proxystore_ready(self) -> bool:
        """Returns a boolean for whether the given `Topology` has the necessary data
        on each node to use ProxyStore.

        Returns:
            `True` if the given `Topology` has all information to use ProxyStore,
            `False` otherwise.
        """
        for node in self.nodes():
            proxystore_id = node.proxystore_id
            match proxystore_id:
                case UUID():
                    continue
                case str():
                    try:
                        UUID(proxystore_id)
                    except ValueError:
                        return False
                case None:
                    return False

        return True

    ###################################################################################

    @property
    def coordinator(self) -> Node:
        """The *Coordinator* node in the `Topology`.

        Returns:
            The Coordinator node.
        """
        return next(iter(self.nodes(NodeKind.COORDINATOR)))

    @property
    def aggregators(self) -> list[Node]:
        """The *Aggregator* nodes in the `Topology`.

        Returns:
            The Aggregator nodes.
        """
        return self.nodes(NodeKind.AGGREGATOR)

    @property
    def workers(self) -> list[Node]:
        """The *Worker* nodes in the `Topology`.

        Returns:
            The Worker nodes.
        """
        return self.nodes(NodeKind.WORKER)

    ###################################################################################

    @property
    def node_dict(self) -> dict[NodeID, Node]:
        """List of `Node`s in the `Topology`."""
        return self._nodes

    @property
    def edges(self) -> list[Edge]:
        """List of edges between `Node`s that exist in the `Topology`."""
        return self._edges

    @property
    def graph(self) -> nx.DiGraph:
        """Directed graph defining the topology's connectivity."""
        return self._graph

    @property
    def source(self) -> pathlib.Path | None:
        """Indicates the file used to instantiate the `Topology` (if one was used)."""
        return self._source

    ###################################################################################

    @classmethod
    def from_adj_list(cls, adj_list: t.Mapping[NodeID, t.Iterable[NodeID]]) -> Topology:
        """
        Adjacency list factory method based on
        [`from_adj_list`][flight.system.io.from_adj_list].
        """
        return cls(*from_adj_list(adj_list))

    @classmethod
    def from_adj_matrix(cls, matrix: ArrayLike) -> Topology:
        """
        Adjacency matrix factory method based on
        [`from_adj_matrix`][flight.system.io.from_adj_matrix].
        """
        return cls(*from_adj_matrix(matrix))

    @classmethod
    def from_dict(cls, data: GraphDict) -> Topology:
        """
        Dictionary factory method based on
        [`from_dict`][flight.system.io.from_dict].
        """
        return cls(*from_dict(data))

    @classmethod
    def from_edgelist(cls, edgelist: list[Edge]) -> Topology:
        raise NotImplementedError

    @classmethod
    def from_json(cls, path: pathlib.Path | str, safe_load: bool = True) -> Topology:
        """
        JSON factory method based on
        [`from_json`][flight.system.io.from_json].
        """
        return cls(*from_json(path, safe_load))

    @classmethod
    def from_networkx(cls, graph: nx.DiGraph) -> Topology:
        """
        NetworkX factory method based on
        [`from_networkx`][flight.system.io.from_networkx].
        """
        return cls(*from_networkx(graph))

    @classmethod
    def from_yaml(cls, path: pathlib.Path | str) -> Topology:
        """
        YAML factory method based on
        [`from_yaml`][flight.system.io.from_yaml].
        """
        return cls(*from_yaml(path))


def resolve_node_or_idx(node_or_idx: Node | NodeID) -> NodeID:
    """
    Returns the `NodeID` from an ambiguous input that can either
    be a `Node` or a `NodeID`.

    Args:
        node_or_idx (Node | NodeID): Data of either type `Node` or `NodeID`.

    Returns:
        The `NodeID`.

    Throws:
        - `ValueError`: Illegal value is passed in as input.
    """
    if isinstance(node_or_idx, Node):
        return node_or_idx.idx
    elif isinstance(node_or_idx, NodeID):
        return node_or_idx
    else:
        raise ValueError("Argument `node_or_idx` must be of type `Node` or `NodeID`.")
