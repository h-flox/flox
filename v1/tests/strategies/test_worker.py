import typing as t

from v1.flight import WorkerStrategy
from v1.flight import DefaultWorkerStrategy

if t.TYPE_CHECKING:
    NodeState: t.TypeAlias = t.Any


def test_instance():
    """Test that the associated node strategy type follows the correct protocols."""
    default_worker = DefaultWorkerStrategy()

    assert isinstance(default_worker, WorkerStrategy)
