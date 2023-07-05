import copy
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from lightning import LightningModule

from flox.worker.core import work


class LocalExec(ABC):
    def __init__(self, max_workers: int = 1):
        self.max_workers = max_workers

    def submit_jobs(
            self,
            workers: dict,
            logic,
            module: LightningModule,
            **kwargs
    ) -> list[Future]:
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as _exec:
            for worker_id, worker in workers.items():
                fut = _exec.submit(
                    work,
                    worker_id=worker_id,
                    logic=logic,
                    module=copy.deepcopy(module),
                    **kwargs
                )
                futures.append(fut)
        return futures
