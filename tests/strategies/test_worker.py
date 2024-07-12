from flight.strategies import WorkerStrategy
from flight.strategies.base import DefaultWorkerStrategy


def test_instance():
    default_worker = DefaultWorkerStrategy()

    assert isinstance(default_worker, WorkerStrategy)
