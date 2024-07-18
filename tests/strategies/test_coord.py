import pytest
from numpy.random import default_rng

from flight.strategies import CoordStrategy
from flight.strategies.base import DefaultCoordStrategy
from flight.strategies.commons.worker_selection import random_worker_selection
from tests.strategies.environment import create_children


def test_instance():
    """Test that the associated node strategy type follows the correct protocols."""
    default_coord = DefaultCoordStrategy()

    assert isinstance(default_coord, CoordStrategy)


def test_worker_selection():
    """Tests both fix and probabilistic worker selection on five workers."""
    gen = default_rng()
    for _ in range(5):
        children = create_children(numWorkers=5)
        # fixed random
        fixed_random = random_worker_selection(
            children,
            participation=1,
            probabilistic=False,
            always_include_child_aggregators=True,
            rng=gen,
        )
        # prob random
        prob_random = random_worker_selection(
            children,
            participation=1,
            probabilistic=True,
            always_include_child_aggregators=True,
            rng=gen,
        )

    for child in children:
        assert child in fixed_random and child in prob_random


class TestInvalidFixedSelection:
    def test_fixed_random(self):
        """Tests an invalid level of participation on fixed selection raises a 'ValueError'"""
        gen = default_rng()

        children = create_children(numWorkers=5)

        with pytest.raises(ValueError):
            fixed_random = random_worker_selection(
                children,
                participation=2,
                probabilistic=False,
                always_include_child_aggregators=True,
                rng=gen,
            )

    def test_fixed_random_mix(self):
        """Tests an invalid level of participation on prob selection raises a 'ValueError'"""
        gen = default_rng()

        children = create_children(numWorkers=1, numAggr=2)

        with pytest.raises(ValueError):
            fixed_random = random_worker_selection(
                children,
                participation=2,
                probabilistic=False,
                always_include_child_aggregators=True,
                rng=gen,
            )
