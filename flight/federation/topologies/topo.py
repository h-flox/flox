from __future__ import annotations

import enum
import functools
import pathlib
import typing as t
import uuid

import networkx as nx

import flight.federation.topologies.io as io

from .exceptions import NodeNotFoundError, TopologyException
from .node import Node, NodeID, NodeKind
from .types import GraphDict

if t.TYPE_CHECKING:
    import numpy as np


def resolve_node_or_idx(node_or_idx: Node | NodeID) -> NodeID:
    if isinstance(node_or_idx, Node):
        return node_or_idx.idx
    elif isinstance(node_or_idx, NodeID):  # type: ignore # (mypy wants `int | str`)
        return node_or_idx
    else:
        raise ValueError("Argument `node_or_idx` must be of type `Node` or `NodeID`.")


class TopologyKind(enum.Enum):
    HUB_SPOKE = enum.auto()  # two-tier, star network
    COMPLEX = enum.auto()


class Topology:
    _nodes: t.Mapping[NodeID, Node]
    _edges: list[tuple[NodeID, NodeID]]
    _source: pathlib.Path | None
    _graph: nx.DiGraph

    def __init__(
        self,
        nodes: list[Node],
        edges: list[tuple[NodeID, NodeID]],
        source: pathlib.Path | str | None = None,
    ):
        node_dict = {node.idx: node for node in nodes}
        graph = nx.DiGraph()
        graph.add_nodes_from(list(node_dict))
        graph.add_edges_from(edges)
        # validate_graph(node_dict.values(), edges, graph)

        self._nodes = node_dict
        self._edges = edges
        self._graph = graph

        match source:
            case pathlib.Path():
                self._source = source
            case str():
                self._source = pathlib.Path(source)
            case None:
                self._source = source
            case _:
                raise ValueError(
                    "Illegal argument kind for `source`. Tip: This should not be "
                    "set by the user. Leave it to the constructor methods."
                )

        validate(self)

    def __contains__(self, idx: NodeID) -> bool:
        """
        Returns `True` if a node with the given ID is in the topology;
        `False` otherwise.
        """
        return idx in self._nodes

    def __getitem__(self, idx: NodeID) -> Node:
        """
        Get the `Node` of the topology by its ID.

        Args:
            idx (NodeID): The ID of the node to return.

        Throws:
            - `NodeNotFound` in the event that the given `NodeID` is not part of
              the topology.

        Returns:
            The requested node.
        """
        if idx not in self:
            raise NodeNotFoundError()
        return self._nodes[idx]

    def __iter__(self) -> t.Iterator[NodeID]:
        """Iterates through all the node IDs in the Topology."""
        return iter(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def nodes(self, kind: NodeKind | str | None = None) -> t.Iterator[Node]:
        """
        Creates an iterator that loops through nodes making up the topology.

        The optional parameter, `kind`, can be used to filter only certain nodes to be
        included in the returned iterator.

        Args:
            kind (NodeKind | str | None): The kind of nodes to include in the iterator.
                If `None`, then all nodes in the topology are included in the returned
                iterator.

        Raises:
            - `ValueError` in the event the user provides an illegal `str` for arg
              `kind` (see docs for `NodeKind` enum).

        Examples:
            >>> nodes: list[Node] = ...
            >>> edges: list[tuple[NodeID, NodeID]] = ...
            >>> topo = Topology(nodes, edges)
            >>> for node in topo.nodes(kind="coordinator"):
            >>>     print(node)
            idx=0 kind=<NodeKind.COORD: 'coordinator'> globus_comp_id=None
            proxystore_id=None extra=None

        Returns:
            An iterator through the nodes.
        """
        if isinstance(kind, str):
            kind = NodeKind(kind)

        for idx in self:
            if kind is None or self[idx].kind == kind:
                yield self._nodes[idx]

    def parent(self, node_or_idx: Node | NodeID) -> Node:
        """
        Returns the parent of the given node.

        Args:
            node_or_idx (Node | NodeID): The `Node` whose parent we wish to return.
                The given parameter can be either of type `Node` or `NodeID`. Any other
                value type will result is a `ValueError`.

        Throws:
            - `TopologyException` is raised in the event that the user requests the
              parent of the coordinator.
            - `ValueError` in the event that the user provides a parameter value that
              is neither a `Node` nor `NodeID`.
            - `NodeNotFound` in the event that the given `Node` or `NodeID` is not
              part of the topology.

        Returns:
            The parent node of the given node.
        """
        idx = resolve_node_or_idx(node_or_idx)
        if idx not in self:
            raise NodeNotFoundError()

        node = self[idx]
        if node.kind is NodeKind.COORD:
            raise TopologyException(f"{node.kind.value.title()} node has no parent.")

        parent = next(iter(self._graph.predecessors(idx)))
        return self[parent]

    def children(self, node_or_idx: Node | NodeID) -> t.Iterator[Node]:
        """
        Gets an iterator that loops through the children of the given node.

        Args:
            node_or_idx (Node | NodeID): The `Node` whose children we wish to return.
                The given parameter can be either of type `Node` or `NodeID`. Any other
                value type will result is a `ValueError`.

        Throws:
            - `ValueError` in the event that the user provides a parameter value that
              is neither a `Node` nor `NodeID`.
            - `NodeNotFound` in the event that the given `Node` or `NodeID` is not part
              of the topology.

        Returns:
            Iterator that loops through the children of the given Node.
        """
        idx = resolve_node_or_idx(node_or_idx)
        for child in self._graph.successors(idx):
            yield self[child]

    def number_of_nodes(self, kind: NodeKind | str | None = None):
        """
        Returns the number of nodes in the network (of a designated kind if specified).

        Args:
            kind (NodeKind | str | None): Specifies the kind of nodes to count. This is
                `None` by default, which will return the total number of nodes in the
                Topology.

        Notes:
            If you wish to use no argument (i.e., `topo.number_of_nodes()`), it is
            recommended that you instead use `len()`. The `number_of_nodes()` method
            iterates through the nodes, whereas the implementation for `len()` does not.

        Returns:
            The number of nodes of a specified kind (if one is given).
        """
        return sum(1 for _ in self.nodes(kind))

    @functools.cached_property
    def coordinator(self) -> Node:
        """Returns the *Coordinator* node in the Topology."""
        return next(iter(self.nodes(NodeKind.COORD)))

    @property
    def aggregators(self) -> t.Iterator[Node]:
        """Iterator that loops through the *Aggregator* nodes in the Topology."""
        return self.nodes(NodeKind.AGGR)

    @property
    def workers(self) -> t.Iterator[Node]:
        """Iterator that loops through the *Worker* nodes in the Topology."""
        return self.nodes(NodeKind.WORKER)

    @functools.cached_property
    def proxystore_ready(self) -> bool:
        for node in self.nodes():
            proxystore_id = node.proxystore_id
            match proxystore_id:
                case uuid.UUID():
                    continue
                case None:
                    return False
                case str():
                    try:
                        uuid.UUID(proxystore_id)
                    except ValueError:
                        return False
        return True

    @functools.cached_property
    def globus_ready(self) -> bool:
        for node in self.nodes():
            if node.kind is NodeKind.COORD:
                continue
            globus_id = node.globus_comp_id
            if any([globus_id is None, isinstance(globus_id, uuid.UUID) is False]):
                return False
        return True

    @functools.cached_property
    def kind(self) -> TopologyKind:
        # TODO: Finish the implementation of this.
        return TopologyKind.HUB_SPOKE

    @classmethod
    def from_adj_list(
        cls,
        adj_list: t.Mapping[NodeID, t.Sequence[NodeID]],
    ) -> Topology:
        """
        Creates a Topology instance using
        [`from_adj_list`][flight.federation.topologies.io.from_adj_list].

        Args:
            adj_list (t.Mapping[NodeID, t.Sequence[NodeID]]): Adjacency list.

        Returns:
            A `Topology` instance.
        """
        return cls(*io.from_adj_list(adj_list))

    @classmethod
    def from_adj_matrix(cls, adj_matrix: np.ndarray) -> Topology:
        """
        Creates a Topology instance using
        [`from_adj_matrix`][flight.federation.topologies.io.from_adj_matrix].

        Args:
            adj_matrix (np.ndarray): Adjacency matrix.

        Returns:
            A `Topology` instance.
        """
        return cls(*io.from_adj_matrix(adj_matrix))

    @classmethod
    def from_dict(cls, data: GraphDict) -> Topology:
        """
        Creates a Topology instance from a `dict` using
        [`from_dict`][flight.federation.topologies.io.from_dict].

        Args:
            data (GraphDict): A dictionary where each top-level key is the node ID
                (`str` or `int`) and the values are Mappings (e.g., dicts) with `str`
                keys for each input into the `Node` class and the child Node IDs.

        Returns:
            A `Topology` instance.
        """
        return cls(*io.from_dict(data))

    @classmethod
    def from_edgelist(cls, path: pathlib.Path | str) -> Topology:
        """
        Creates a Topology instance using
        [`from_edgelist`][flight.federation.topologies.io.from_edgelist].

        Args:
            path (pathlib.Path | str): Path to edgelist.

        Returns:
            A `Topology` instance.
        """
        return Topology(*io.from_edgelist(path))

    @classmethod
    def from_json(cls, path: pathlib.Path | str, safe_load: bool = True) -> Topology:
        """
        Creates a Topology instance from a `*.json` file using
        [`from_adj_matrix`][flight.federation.topologies.io.from_adj_matrix].

        Args:
            path (pathlib.Path | str): Path to the JSON file.
            safe_load (bool): Will convert node IDs (i.e., top-level keys and
                children IDs) to strings. Should be set to `True` (default) in most
                circumstances.

        Returns:
            A `Topology` instance.
        """
        return cls(*io.from_json(path, safe_load=safe_load))

    @classmethod
    def from_networkx(cls, graph: nx.DiGraph) -> Topology:
        """
        Creates a `Topology` instance from a NetworkX directed graph using the utility
        function [`from_nx`][flight.federation.topologies.io.from_nx].


        Args:
            graph (nx.DiGraph): NetworkX directed graph to convert into a Topology.

        Notes:
            Any node-specific features (e.g., Globus Compute UUID) should be stored as
            attributes on the node itself. Anything besides the standard `Node`
            features should be stored in `extra`.

        Throws:
            - `ValueError` if an undirected `nx.Graph` object is passed instead of a
              directed `nx.DiGraph`.

        Returns:
            A `Topology` instance.
        """
        return Topology(*io.from_nx(graph))

    @classmethod
    def from_nx(cls, graph: nx.DiGraph) -> Topology:
        """Alias for `Topology.from_networkx()`."""
        return cls.from_networkx(graph)

    @classmethod
    def from_yaml(cls, path: pathlib.Path | str) -> Topology:
        """
        Creates a `Topology` instance from a YAML file.

        Args:
            path (pathlib.Path | str): Path to the YAML file.

        Notes:
            Leverages the implementation of `from_dict`, so be sure to structure the
            YAML file in a way similar to what is expected of a dictionary formatted
            as a `GraphDict`.

        Returns:
            A `Topology` instance.
        """
        return cls(*io.from_yaml(path))


def validate(topo: Topology) -> None:
    """
    Validates whether the provided topology is structurally legal or not. If the
    topology is not structurally legal, then `TopologyException` is thrown. If no
    exception is thrown, then the topology is legal.

    Args:
        topo (Topology): The `Topology` instance to validate.

    Throws:
        - `TopologyException` if an illegal topology has been defined based on Nodes,
          edges/links, and underlying graph. Exception messages will more explicitly
          state the exact issue. Refer to the docs for more information about the
          requirements for a legal Flight topology.
    """
    # noinspection PyProtectedMember
    graph: nx.DiGraph = topo._graph
    # noinspection PyProtectedMember
    nodes: t.Mapping[NodeID, Node] = topo._nodes
    # edges: list[NodeLink] = topo._edges

    if not isinstance(graph, nx.DiGraph):
        raise TopologyException(
            "Graphs for a legal `Topology` must be *directed* (i.e., `nx.DiGraph`)."
        )

    if not nx.is_tree(graph):
        raise TopologyException(
            "Graph structure of a legal `Topology` must be a *tree* (i.e., a "
            "connected graph with no  cycles)."
        )

    num_kinds = {NodeKind(kind): 0 for kind in NodeKind}
    for node in nodes.values():
        num_kinds[node.kind] += 1

    if num_kinds[NodeKind.COORD] != 1:
        raise TopologyException(
            "Legal Flight topologies must have exactly *one* "
            "Coordinator node (of kind `NodeKind.COORD`)."
        )

    if num_kinds[NodeKind.WORKER] == 0:
        raise TopologyException(
            "Legal Flight topologies must have at least 1 Worker node "
            "(of kind `NodeKind.WORKER`)."
        )

    for node_idx in graph.nodes():
        node = topo[node_idx]
        kind_str = node.kind.title()

        num_parents = len(list(graph.predecessors(node_idx)))
        num_children = len(list(graph.successors(node_idx)))

        match node.kind:
            case NodeKind.COORD:
                if not num_parents == 0:
                    raise TopologyException(f"{kind_str} node cannot have a parent.")
                if not num_children >= 1:
                    raise TopologyException(
                        f"{kind_str} node must have at least 1 child."
                    )
            case NodeKind.AGGR:
                if not num_parents == 1:
                    raise TopologyException(
                        f"{kind_str} node(s) must have exactly 1 parent."
                    )
                if not num_children >= 1:
                    raise TopologyException(
                        f"{kind_str} node(s) must have at least 1 child."
                    )
            case NodeKind.WORKER:
                if not num_parents == 1:
                    raise TopologyException(
                        f"{kind_str} node(s) must have exactly 1 parent."
                    )
                if not num_children == 0:
                    raise TopologyException(
                        f"{kind_str} node(s) cannot have any children nodes."
                    )
