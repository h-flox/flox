from __future__ import annotations

import json
import typing as t

import networkx as nx
import pytest
import yaml

from flight.federation.topologies.node import NodeID, NodeKind
from flight.federation.topologies.topo import Topology
from flight.federation.topologies.types import GraphDict

from .fixtures import three_tier_graph, two_tier_graph  # noqa: F401


def subtest(topo: Topology, num_nodes: int, num_aggrs: int, num_workers: int):
    assert isinstance(topo, Topology)
    assert len(topo) == num_nodes

    assert topo.number_of_nodes(NodeKind.COORD) == 1
    assert topo.number_of_nodes(NodeKind.AGGR) == num_aggrs
    assert topo.number_of_nodes(NodeKind.WORKER) == num_workers

    assert topo.number_of_nodes(NodeKind.COORD) == topo.number_of_nodes("coordinator")
    assert topo.number_of_nodes(NodeKind.AGGR) == topo.number_of_nodes("aggregator")
    assert topo.number_of_nodes(NodeKind.WORKER) == topo.number_of_nodes("worker")


class TestDictionaryIO:
    def test_from_dict_1(self, two_tier_graph):  # noqa: F811
        """Tests loading a simple, hub-spoke topology from a dict."""
        graph, stats = two_tier_graph
        topo = Topology.from_dict(graph)
        subtest(topo, **stats)

        for node, node_data in graph.items():
            children = node_data["children"]
            for child in children:
                assert topo.parent(child).idx == node

    def test_from_dict_2(self, three_tier_graph):  # noqa: F811
        """Tests loading a simple, hub-spoke topology from a dict."""
        graph, stats = three_tier_graph
        topo = Topology.from_dict(graph)
        subtest(topo, **stats)

        for node, node_data in graph.items():
            children = node_data["children"]
            for child in children:
                assert topo.parent(child).idx == node


class TestJsonIO:
    @staticmethod
    def child_subtest(topo: Topology, graph: GraphDict):
        for node, node_data in graph.items():
            children = node_data["children"]
            for child in children:
                child = str(child)
                assert topo.parent(child).idx == node

    def test_from_json_1(self, tmp_path, two_tier_graph):  # noqa: F811
        graph, stats = two_tier_graph
        filename = "tmp-1.json"

        with open(tmp_path / filename, "w") as fp:
            json.dump(graph, fp)
        topo = Topology.from_json(tmp_path / filename)
        subtest(topo, **stats)

        with open(tmp_path / filename) as fp:
            json_graph = json.load(fp)
        TestJsonIO.child_subtest(topo, json_graph)

    def test_from_json_2(self, tmp_path, three_tier_graph):  # noqa: F811
        """Tests loading a simple, hub-spoke topology from a dict."""

        graph, stats = three_tier_graph
        filename = "tmp-2.json"

        with open(tmp_path / filename, "w") as fp:
            json.dump(graph, fp)
        topo = Topology.from_json(tmp_path / filename, safe_load=True)
        subtest(topo, **stats)

        with open(tmp_path / filename) as fp:
            json_graph = json.load(fp)
        TestJsonIO.child_subtest(topo, json_graph)


class TestNetworkXIO:
    @staticmethod
    def _assign_roles(
        graph: nx.Graph | nx.DiGraph,
        coord: NodeID,
        aggrs: t.Iterable[NodeID],
        workers: t.Iterable[NodeID],
    ):
        graph.nodes[coord]["kind"] = "coordinator"
        for i in aggrs:
            graph.nodes[i]["kind"] = "aggregator"
        for i in workers:
            graph.nodes[i]["kind"] = "worker"

    def test_from_nx_1(self):
        """Tests how the I/O of loading a 2-tier NetworkX graph using a star graph topology."""
        num_nodes = 11

        star_graph = nx.star_graph(num_nodes - 1)
        coord = 0
        aggrs = ()
        workers = range(1, num_nodes)

        TestNetworkXIO._assign_roles(star_graph, coord, aggrs, workers)
        with pytest.raises(ValueError):
            Topology.from_networkx(star_graph)

        graph = nx.DiGraph()
        graph.add_nodes_from(list(star_graph.nodes()))
        graph.add_edges_from(list(star_graph.edges()))
        TestNetworkXIO._assign_roles(graph, coord, aggrs, workers)
        topo = Topology.from_networkx(graph)

        subtest(topo, num_nodes=num_nodes, num_aggrs=0, num_workers=num_nodes - 1)
        assert topo._graph.number_of_edges() == num_nodes - 1

    def test_from_nx_2(self):
        """Tests how I/O of loading a hierarchical NetworkX graph using a balanced tree topology."""
        branching_factor = 5
        height = 2
        tree = nx.balanced_tree(branching_factor, height)

        coord = 0
        aggrs = range(1, 6)
        workers = range(6, tree.number_of_nodes())

        TestNetworkXIO._assign_roles(tree, coord, aggrs, workers)
        with pytest.raises(ValueError):
            Topology.from_networkx(tree)

        graph = nx.DiGraph()
        graph.add_nodes_from(list(tree.nodes()))
        graph.add_edges_from(list(tree.edges()))
        TestNetworkXIO._assign_roles(graph, coord, aggrs, workers)
        topo = Topology.from_networkx(graph)

        subtest(topo, num_nodes=tree.number_of_nodes(), num_aggrs=5, num_workers=25)
        assert topo._graph.number_of_edges() == tree.number_of_edges()


class TestYamlIO:
    @staticmethod
    def child_subtest(topo: Topology, graph: GraphDict):
        for node, node_data in graph.items():
            children = node_data["children"]
            for child in children:
                assert topo.parent(child).idx == node

    def test_from_yaml_1(self, tmp_path, two_tier_graph):  # noqa: F811
        graph, stats = two_tier_graph
        filename = "tmp-1.yaml"

        with open(tmp_path / filename, "w") as fp:
            yaml.dump(graph, fp)
        topo = Topology.from_yaml(tmp_path / filename)
        subtest(topo, **stats)

        with open(tmp_path / filename) as fp:
            yaml_graph = yaml.safe_load(fp)
        TestYamlIO.child_subtest(topo, yaml_graph)

    def test_from_yaml_2(self, tmp_path, three_tier_graph):  # noqa: F811
        """Tests loading a simple, hub-spoke topology from a dict."""

        graph, stats = three_tier_graph
        filename = "tmp-2.yaml"

        with open(tmp_path / filename, "w") as fp:
            yaml.dump(graph, fp)
        topo = Topology.from_yaml(tmp_path / filename)
        subtest(topo, **stats)

        with open(tmp_path / filename) as fp:
            yaml_graph = yaml.safe_load(fp)
        TestYamlIO.child_subtest(topo, yaml_graph)
