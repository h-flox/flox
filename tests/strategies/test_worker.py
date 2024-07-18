import typing as t

import torch

from flight.federation.jobs.result import Result
from flight.strategies import WorkerStrategy
from flight.strategies.base import DefaultWorkerStrategy

if t.TYPE_CHECKING:
    NodeState: t.TypeAlias = t.Any


def test_instance():
    default_worker = DefaultWorkerStrategy()

    assert isinstance(default_worker, WorkerStrategy)


def test_default_methods():
    default_worker = DefaultWorkerStrategy()
    workerstate = ("NodeID 1", "num_data_samples 2")
    data = {
        "train/acc": torch.tensor(0.3, dtype=torch.float32),
        "train/loss": torch.tensor(0.7, dtype=torch.float32),
    }
    # result = Result(workerstate, 1, data, [], {})
    # optimizer = torch.optim.Optimizer(data.values(), {})

    # assert default_worker.start_work(workerstate) == workerstate
    # assert default_worker.before_training(workerstate, data) == (workerstate, data)
    # assert default_worker.after_training(workerstate, optimizer) == workerstate
    # assert default_worker.end_work(result) == result
