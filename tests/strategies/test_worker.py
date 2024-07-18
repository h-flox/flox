import typing as t

import torch

from flight.federation.jobs.result import Result
from flight.strategies import WorkerStrategy
from flight.strategies.base import DefaultWorkerStrategy

if t.TYPE_CHECKING:
    NodeState: t.TypeAlias = t.Any


def test_instance():
    """Test that the associated node strategy type follows the correct protocols."""
    default_worker = DefaultWorkerStrategy()

    assert isinstance(default_worker, WorkerStrategy)
