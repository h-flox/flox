import pytest
from numpy.random import default_rng

from flight.federation.topologies.node import Node, NodeKind
from flight.strategies import CoordStrategy
from flight.strategies.base import DefaultCoordStrategy
from flight.strategies.commons.worker_selection import random_worker_selection


def test_instance():
    default_coord = DefaultCoordStrategy()

    assert isinstance(default_coord, CoordStrategy)


def test_worker_selection():
    gen = default_rng()

    worker1 = Node(idx=1, kind=NodeKind.WORKER)
    worker2 = Node(idx=2, kind=NodeKind.WORKER)
    workers = [worker1, worker2]
    # fixed random
    fixed_random = random_worker_selection(
        workers,
        participation=1,
        probabilistic=False,
        always_include_child_aggregators=True,
        rng=gen,
    )
    # prob random
    prob_random = random_worker_selection(
        workers,
        participation=1,
        probabilistic=True,
        always_include_child_aggregators=True,
        rng=gen,
    )

    for worker in workers:
        assert worker in fixed_random and worker in prob_random


class TestInvalidFixedSelection:
    def test_fixed_random(self):
        gen = default_rng()

        workers = [Node(idx=i, kind=NodeKind.WORKER) for i in range(1, 6)]

        with pytest.raises(ValueError):
            fixed_random = random_worker_selection(
                workers,
                participation=2,
                probabilistic=False,
                always_include_child_aggregators=True,
                rng=gen,
            )

    def test_fixed_random_mix(self):
        gen = default_rng()

        children = [Node(idx=i, kind=NodeKind.AGGR) for i in range(1, 3)]
        children.append(Node(idx=3, kind=NodeKind.WORKER))

        with pytest.raises(ValueError):
            fixed_random = random_worker_selection(
                children,
                participation=2,
                probabilistic=False,
                always_include_child_aggregators=True,
                rng=gen,
            )
