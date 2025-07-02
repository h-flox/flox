from __future__ import annotations

import pytest

from flight.system.topology import Topology
from flight.system.utils import hierarchical_topology


def test_from_adjacency_matrix():
    topo = Topology.from_adj_matrix(
        [
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    assert isinstance(topo, Topology)
    assert len(topo) == 3
    assert topo.number_of_nodes("worker") == 2
    assert topo.number_of_nodes("coordinator") == 1
    assert topo.number_of_nodes() == len(topo)


def test_hierarchical_topo_fn():
    n = 20
    rng = 42

    # Check the `None` case.
    topo = hierarchical_topology(n, aggr_shape=None, rng=rng)
    assert topo.number_of_nodes() == n + 1  # +1 for the coordinator
    assert topo.number_of_nodes("coordinator") == 1
    assert topo.number_of_nodes("worker") == n
    assert topo.height == 1

    # Check the ascending `aggr_shape` case.
    aggr_shape = []
    for i in range(2, 6):
        aggr_shape.append(i)
        topo = hierarchical_topology(n, aggr_shape=aggr_shape, rng=rng)
        assert topo.number_of_nodes() == n + sum(aggr_shape) + 1
        assert topo.number_of_nodes("coordinator") == 1
        assert topo.number_of_nodes("aggregator") == sum(aggr_shape)
        assert topo.number_of_nodes("worker") == n
        assert topo.height == len(aggr_shape) + 1

        with pytest.raises(ValueError):
            # This should raise an error because there should not be fewer workers
            # than aggregators (in the bottom-most tier).
            hierarchical_topology(1, aggr_shape=aggr_shape, rng=rng)

    # Check the descending `aggr_shape` case.
    aggr_shape = [5]
    for i in range(4, 0, -1):
        aggr_shape.append(i)
        with pytest.raises(ValueError):
            # This should raise an error because the shape is not ascending.
            hierarchical_topology(n, aggr_shape=aggr_shape, rng=rng)
