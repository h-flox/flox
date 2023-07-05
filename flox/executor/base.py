from abc import ABC, abstractmethod
from concurrent.futures import Future
from lightning import LightningModule


class BaseExec(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def submit_jobs(
            self,
            workers: dict,
            logic,
            module: LightningModule,
            **kwargs
    ) -> list[Future]:
        pass
