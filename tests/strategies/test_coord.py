from flight.strategies import CoordStrategy
from flight.strategies.base import DefaultCoordStrategy

def test_instance():
    default_coord = DefaultCoordStrategy()

    assert isinstance(default_coord, CoordStrategy)

