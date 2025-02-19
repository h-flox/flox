import pytest

from v1.flight import Topology
from v1.flight import flat_topology, hierarchical_topology


class TestFlatTopologies:
    def test_valid_flat_topos(self):
        for n in range(1, 10 + 1):
            topo = flat_topology(n)
            assert isinstance(topo, Topology)
            assert topo.number_of_nodes() == n + 1
            assert len(list(topo.workers)) == n
            for worker in topo.workers:
                print(worker.extra)
                assert len(worker.extra) == 0

        topo = flat_topology(10, foo="bar")
        for worker in topo.workers:
            assert worker["foo"] == "bar"

    def test_invalid_flat_topos(self):
        for n in [-2, -1, 0]:
            with pytest.raises(ValueError):
                _ = flat_topology(n)


class TestHierarchicalTopologies:
    def test_valid_hier_topos(self):
        topo = hierarchical_topology(10, aggr_shape=(2,))
        aggrs = list(topo.aggregators)
        workers = list(topo.workers)
        assert len(aggrs) == 2
        assert len(workers) == 10
