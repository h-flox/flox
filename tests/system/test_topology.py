import pytest

from flight.system.topology import Topology


def test_from_adjacency_matrix():
    A = [[0, 1, 1], [0, 0, 0], [0, 0, 0]]
    topo = Topology.from_adj_matrix(A)
    assert isinstance(topo, Topology)
    assert len(topo) == 3
    assert topo.number_of_nodes("worker") == 2
    assert topo.number_of_nodes("coordinator") == 1
    assert topo.number_of_nodes() == len(topo)
