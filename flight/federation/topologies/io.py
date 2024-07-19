"""
I/O module for loading Flight topologies for several different data formats.

More specifically, the functions in this submodule will return a node-list (`List[Node]`) and an
edge-list (`List[NodeLink]`) which are then given as inputs into the `Topology` class for construction.

It is recommended that users do **not** use the functions of this submodule directly. We instead encourage users
to take advantage of the wrapper functions directly provided as class methods in the `Topology` class. So, if users
wish to use `from_yaml()` to create a Topology with a YAML file, we encourage the use the `Topology.from_yaml()` method
instead.
"""

from __future__ import annotations

import json
import typing as t

import networkx as nx
import yaml

from .node import Node, NodeID, NodeKind

if t.TYPE_CHECKING:
    import pathlib

    import numpy as np

    from .types import GraphDict, NodeLink


def from_adj_list(
    adj_list: t.Mapping[NodeID, t.Sequence[NodeID]]
) -> tuple[list[Node], list[NodeLink]]:
    raise NotImplementedError()


def from_adj_matrix(matrix: np.ndarray) -> tuple[list[Node], list[NodeLink]]:
    raise NotImplementedError()


def from_dict(data: GraphDict) -> tuple[list[Node], list[NodeLink]]:
    """
    Parses a mapping object (e.g., `dict` or `OrderedDict`) and returns its corresponding node list and edge list.

    Args:
        data (GraphDict): A dictionary where each top-level key is the node ID (`str` or `int`) and the
            values are Mappings (e.g., dicts) with `str` keys for each input into the `Node` class and
            the child Node IDs.

    Returns:
        Tuple of two lists: (i) list of `Node`s and (ii) list of `NodeLink`s.
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
    raise NotImplementedError()


def from_json(
    path: pathlib.Path | str, safe_load: bool = True
) -> tuple[list[Node], list[NodeLink]]:
    """
    Loads a JSON file and returns its corresponding node list and edge list.

    Args:
        path (pathlib.Path | str): Path to the JSON file.
        safe_load (bool): Will convert node IDs (i.e., top-level keys and children IDs) to strings.
            Should be set to `True` (default) in most circumstances.

    Notes:
        Any Topology defined as a JSON will require all Node IDs to be of type `str`.
        This is due to a limitation of the JSON format (JSON does not support integer keys).
        As a safety precaution, this function will convert any Node ID to a string if
        `safe_load = True`. But, this is detail is worth being aware of for users.

    Returns:
        Tuple of two lists: (i) list of `Node`s and (ii) list of `NodeLink`s.
    """
    with open(path) as fp:
        data = json.load(fp)

    if safe_load:
        # Because the JSON format does not support integers for keys, we have to convert every
        # value in `children` from integers to strings. This is a limitation of the data format.
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
        Any node-specific features (e.g., Globus Compute UUID) should be stored as attributes on the
        node itself. Anything besides the standard `Node` features should be stored in `extra`.

    Throws:
        - `ValueError` if an undirected `nx.Graph` object is passed instead of a directed `nx.DiGraph`.

    Returns:
        Tuple of two lists: (i) list of `Node`s and (ii) list of `NodeLink`s.
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
    """Alias for `from_networkx()`."""
    return from_networkx(graph)


def from_yaml(path: pathlib.Path | str) -> tuple[list[Node], list[NodeLink]]:
    """

    Args:
        path (pathlib.Path | str):

    Returns:
        Tuple of two lists: (i) list of `Node`s and (ii) list of `NodeLink`s.
    """
    with open(path) as fp:
        data = yaml.safe_load(fp)

    return from_dict(data)
