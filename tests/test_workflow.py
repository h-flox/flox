from __future__ import annotations

import pytest

from flight.system import Topology
from flight.system.utils import flat_topology, hierarchical_topology
from flight.workflow import get_relevant_nodes


########################################################################################


@pytest.fixture
def topo() -> Topology:
    return hierarchical_topology(aggr_shape=(2, 4), n=10)


@pytest.fixture
def flat_topo() -> Topology:
    return flat_topology(5)


@pytest.fixture
def hier_topo_dict() -> Topology:
    return Topology.from_dict(
        {
            0: {"children": [1, 2], "kind": "coordinator"},
            1: {"children": [11, 12, 13], "kind": "aggregator"},
            2: {"children": [21, 22], "kind": "aggregator"},
            11: {"kind": "worker"},
            12: {"kind": "worker"},
            13: {"kind": "worker"},
            21: {"kind": "worker"},
            22: {"kind": "worker"},
        }
    )


########################################################################################


def test_hierarchical_topo(topo, flat_topo, hier_topo_dict):
    relevant_nodes = get_relevant_nodes(hier_topo_dict, [11, 12, 21])
    assert len(relevant_nodes) == 3
    assert set(relevant_nodes[0]) == {1, 2}
    assert set(relevant_nodes[1]) == {11, 12}
    assert set(relevant_nodes[2]) == {21}
