"""
I/O module for loading Flight topologies for several different data formats.

More specifically, the functions in this submodule will return a node-list
(`List[Node]`) and an edge-list (`List[NodeLink]`) which are then given as inputs
into the `Topology` class for construction.

Notes:
    It is recommended that users do **not** use the functions of this submodule
    directly. We instead encourage users to take advantage of the wrapper functions
    directly provided as class methods in the `Topology` class. So, if users wish
    to use `from_yaml()` to create a Topology with a YAML file, we encourage the use
    the `Topology.from_yaml()` method instead.
"""

from __future__ import annotations

import json
import typing as t

import networkx as nx
import numpy as np
import yaml

from .node import Node, NodeID, NodeKind

if t.TYPE_CHECKING:
    import pathlib

    from numpy.typing import ArrayLike

    from .types import GraphDict, NodeLink


def from_adj_list(
    adj_list: t.Mapping[NodeID, t.Iterable[NodeID]]
) -> tuple[list[Node], list[NodeLink]]:
    raise NotImplementedError()


def from_adj_matrix(matrix: ArrayLike[int]) -> tuple[list[Node], list[NodeLink]]:
    """
    Parses an `ArrayLike` object into a node list and edge list.

    Examples of `ArrayLike` objects include `numpy.ndarray`s and nested lists. Refer to
    this [link](https://numpy.org/doc/2.1/reference/typing.html#numpy.typing.ArrayLike)
    for more information on what qualifies as a valid data type.

    The given adjacency matrix has to be connected, form a tree, contain more than
    1 node, and be directed. The coordinator of the topology is inferred by simply
    using the root of the tree (which is given by a topological sorting of the nodes).

    Args:
        matrix (ArrayLike[int]): The adjacency matrix defining the topology.

    Returns:
        Tuple of two lists:

            1. list of `Node`s
            2. list of `NodeLink`s.

    Throws:
        - `ValueError`: This will be thrown in two scenarios:

            1. The number of nodes of the given adjacency matrix is not greater than 1.
            2. The graph from the adjacency matrix does not form a tree. This is checked
               using the `networkx.is_tree()` function.

        - `NetworkXException`: This is thrown when an issue arises from `networkx` when
            performing the topological sorting on the graph. Namely, this will occur
            if a cycle arises. This can happen if the adjacency matrix is defined as
            an undirected network (i.e., the matrix is equivalent to its own transpose).

    Examples:
        >>> # A tree with 3 nodes; the root has 2 children.
        >>> mat = [[0, 1, 1], [0, 0, 0], [0, 0, 0]]
        >>> nodes, edges = from_adj_matrix(mat)
        >>> nodes
        [Node(idx=0, kind=<NodeKind.COORD: 'coordinator'>, globus_comp_id=None,
        proxystore_id=None, extra=None), Node(idx=1, kind=<NodeKind.WORKER: 'worker'>,
        globus_comp_id=None, proxystore_id=None, extra=None), Node(idx=2,
        kind=<NodeKind.WORKER: 'worker'>, globus_comp_id=None, proxystore_id=None,
        extra=None)]
        >>> edges
        [(0, 1), (0, 2)]
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)

    graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    if graph.number_of_nodes() <= 1:
        raise ValueError("Adjacency matrices for graphs must have more than 1 node.")

    if not nx.is_tree(graph):
        raise ValueError(
            "Graph constructed from given adjacency matrix must form a tree."
        )
    try:
        sorting = list(nx.topological_sort(graph))  # noqa
    except nx.exception.NetworkXException:
        raise nx.exception.NetworkXException(
            "Graph contains a cycle or graph changed during iteration. "
            "The Flight IO function `from_adj_matrix` assumes the network is directed. "
            "However, if your adjacency matrix forms an undirected matrix (i.e., "
            "matrix is equal to its transpose), then this will cause an issue. "
        )

    edges = [(u, v) for (u, v) in graph.edges()]
    coord, others = sorting[0], sorting[1:]
    nodes = [Node(idx=coord, kind=NodeKind.COORD)]

    for node in others:
        children = list(graph.successors(node))
        kind = NodeKind.WORKER if len(children) == 0 else NodeKind.AGGR
        nodes.append(Node(idx=node, kind=kind))

    return nodes, edges


def from_dict(data: GraphDict) -> tuple[list[Node], list[NodeLink]]:
    """
    Parses a mapping object (e.g., `dict` or `OrderedDict`) and returns its
    corresponding node list and edge list.

    Args:
        data (GraphDict): A dictionary where each top-level key is the node ID (`str`
            or `int`) and the values are Mappings (e.g., dicts) with `str` keys for
            each input into the `Node` class and the child Node IDs.

    Returns:
        Tuple of two lists:

            1. list of `Node`s
            2. list of `NodeLink`s.
    """
    nodes: list[Node] = []
    edges: list[NodeLink] = []

    for node_idx, node_data in data.items():
        node = Node(
            idx=node_idx,
            kind=NodeKind(node_data["kind"]),
            globus_comp_id=node_data.get("globus_comp_id", None),
            proxystore_id=node_data.get("proxystore_id", None),
            extra=node_data.get("extra", None),
        )
        nodes.append(node)
        for child_idx in node_data.get("children", []):
            edges.append((node_idx, child_idx))

    return nodes, edges


def from_edgelist(path: pathlib.Path | str) -> tuple[list[Node], list[NodeLink]]:
    """
    This function is not yet implemented.
    """
    raise NotImplementedError()


def from_json(
    path: pathlib.Path | str, safe_load: bool = True
) -> tuple[list[Node], list[NodeLink]]:
    """
    Loads a JSON file and returns its corresponding node list and edge list.

    Args:
        path (pathlib.Path | str): Path to the JSON file.
        safe_load (bool): Will convert node IDs (i.e., top-level keys and children IDs)
            to strings. Should be set to `True` (default) in most circumstances.

    Notes:
        Any Topology defined as a JSON will require all Node IDs to be of type `str`.
        This is due to a limitation of the JSON format (JSON does not support integer
        keys). As a safety precaution, this function will convert any Node ID to a
        string if `safe_load = True`. But, this detail is worth being aware of for
        users.

    Returns:
        Tuple of two lists:

            1. list of `Node`s
            2. list of `NodeLink`s.
    """
    with open(path) as fp:
        data = json.load(fp)

    if safe_load:
        # Because the JSON format does not support integers for keys, we have to
        # convert every value in `children` from integers to strings. This is a
        # limitation of the data format.
        data = {str(key): value for key, value in data.items()}
        for key, value in data.items():
            value = map(lambda val: str(val), value["children"])
            data[key]["children"] = list(value)

    return from_dict(data)


def from_networkx(graph: nx.DiGraph) -> tuple[list[Node], list[NodeLink]]:
    """
    Loads a NetworkX directed graph and returns its node list and edge list.

    Args:
        graph (nx.DiGraph): NetworkX directed graph to convert into a Topology.

    Notes:
        Any node-specific features (e.g., Globus Compute UUID) should be stored as
        attributes on the node itself. Anything besides the standard `Node` features
        should be stored in `extra`.

    Throws:
        - `ValueError` if an undirected `nx.Graph` object is passed instead of a
          directed `nx.DiGraph`.

    Returns:
        Tuple of two lists:

            1. list of `Node`s
            2. list of `NodeLink`s.
    """
    if isinstance(graph, nx.Graph) and not isinstance(graph, nx.DiGraph):
        raise ValueError("`from_nx()` only accepts directed graphs (i.e., nx.DiGraph).")

    nodes = []
    edges = []

    for node_idx, node_data in graph.nodes(data=True):
        node = Node(
            idx=node_idx,
            kind=NodeKind(node_data["kind"]),
            globus_comp_id=node_data.get("globus_comp_id", None),
            proxystore_id=node_data.get("proxystore_id", None),
            extra=node_data.get("extra", None),
        )
        nodes.append(node)
        for child_idx in graph.successors(node_idx):
            edges.append((node_idx, child_idx))

    return nodes, edges


def from_nx(graph: nx.DiGraph) -> tuple[list[Node], list[NodeLink]]:
    """
    Alias for [`from_networkx()`][flight.federation.topologies.io.from_networkx].
    """
    return from_networkx(graph)


def from_yaml(path: pathlib.Path | str) -> tuple[list[Node], list[NodeLink]]:
    """
    Reads a `.yaml` file and parses it into a node list and edge list.

    Args:
        path (pathlib.Path | str): Path to the `.yaml` (or `.yml`) file.

    Returns:
        Tuple of two lists:

            1. list of `Node`s
            2. list of `NodeLink`s.
    """
    with open(path) as fp:
        data = yaml.safe_load(fp)

    return from_dict(data)
