import typing as t

import pytest

from flight.federation.topologies.exceptions import TopologyException
from flight.federation.topologies.node import NodeID, NodeKind
from flight.federation.topologies.topo import Topology
from flight.federation.topologies.types import GraphDict
from .fixtures import three_tier_graph, two_tier_graph  # noqa: F401


def all_disjoint(*args: set[NodeID]):
    """
    Helper function for seeing if sets of Node IDs have any overlapping IDs.

    Args:
        *args (set[NodeID]): A variable number of sets of Node IDs.

    Examples:
        >>> a = {1, 2, 3}
        >>> b = {4, 5, 6}
        >>> c = {7, 4, 6} # 4 and 6 overlap with `b`
        >>> all_disjoint(a, b, c)
        False
        >>> a = {1, 2, 3}
        >>> b = {4, 5, 6}
        >>> c = {7, 8, 9}
        >>> all_disjoint(a, b, c)
        True

    Returns:
        `True` if the sets are all disjoint from each other; `False` otherwise.
    """
    for set_i in args:
        for set_j in args:
            if set_i is set_j:
                continue
            if not set_i.isdisjoint(set_j):
                return False
    return True


def copy_graph(graph: GraphDict) -> dict[NodeID, dict[str, t.Any]]:
    d = {}
    for key, val in graph.items():
        d[key] = {_key: _val for _key, _val in val.items()}
    return d


def change_roles(
    data: GraphDict,
    coords: t.Iterable[NodeID],
    aggrs: t.Iterable[NodeID],
    workers: t.Iterable[NodeID],
) -> GraphDict:
    coords = set(coords) if not isinstance(coords, set) else coords
    aggrs = set(aggrs) if not isinstance(aggrs, set) else aggrs
    workers = set(workers) if not isinstance(workers, set) else workers

    if not all_disjoint(coords, aggrs, workers):
        raise ValueError(
            "Overlapping node IDs for `change_roles` helper function in "
            "`TestInvalidTopologies` test."
        )

    new_graph_dict = {}
    for node, node_data in data.items():
        new_node_data = {key: value for key, value in node_data.items()}
        if node in coords:
            new_node_data["kind"] = NodeKind.COORD
        elif node in aggrs:
            new_node_data["kind"] = NodeKind.AGGR
        elif node in workers:
            new_node_data["kind"] = NodeKind.WORKER
        new_graph_dict[node] = new_node_data

    return new_graph_dict


class TestValidTopologies:
    def test_valid_topo_1(self, two_tier_graph):  # noqa: F811
        graph_dict, stats = two_tier_graph
        try:
            topo = Topology.from_dict(graph_dict)
            assert isinstance(topo, Topology)
        except TopologyException as exc:
            assert False, f"`valid_topo_1` raised {exc}"


class TestInvalidTopologies:
    def test_all_same_kind_topo(self, two_tier_graph):  # noqa: F811
        graph, stats = two_tier_graph
        num_nodes = stats["num_nodes"]
        all_nodes = set(range(num_nodes))

        all_coord_graph = change_roles(graph, all_nodes, {}, {})
        with pytest.raises(TopologyException):
            Topology.from_dict(all_coord_graph)

        all_aggr_graph = change_roles(graph, {}, all_nodes, {})
        with pytest.raises(TopologyException):
            Topology.from_dict(all_aggr_graph)

        all_worker_graph = change_roles(graph, {}, {}, all_nodes)
        with pytest.raises(TopologyException):
            Topology.from_dict(all_worker_graph)

    def test_extra_invalid_edge_1(self, two_tier_graph):  # noqa: F811
        graph = copy_graph(two_tier_graph.graph)
        graph[10]["children"].append(0)
        with pytest.raises(TopologyException):
            Topology.from_dict(graph)

    def test_extra_invalid_edge_2(self, three_tier_graph):  # noqa: F811
        graph = copy_graph(three_tier_graph.graph)
        graph[8]["children"].append(0)
        with pytest.raises(TopologyException):
            Topology.from_dict(graph)
