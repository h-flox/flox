from abc import ABC
from typing import Any, NewType

from depr.worker.base import AbstractWorkerLogic

WorkerID = NewType("WorkerID", str)


class WorkerModule(ABC):
    workers: dict[WorkerID, AbstractWorkerLogic]

    def __init__(self):
        self.workers: dict
