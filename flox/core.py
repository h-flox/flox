import lightning as L
import random

from typing import Type

from flox.typing import WorkerID
from flox.worker import WorkerLogicInterface


def fork_module(
        module: L.LightningModule
) -> L.LightningModule:
    cls = module.__class__
    forked_module = cls()
    forked_module.load_state_dict(module.state_dict())
    return forked_module


def create_workers(
        num: int,
        worker_logic: Type[WorkerLogicInterface]
) -> dict[WorkerID, WorkerLogicInterface]:
    workers = {}
    for idx in range(num):
        n_samples = random.randint(50, 250)
        indices = random.sample(range(60_000), k=n_samples)
        workers[f"Worker-{idx}"] = worker_logic(idx=idx, indices=list(indices))
    return workers
