from abc import ABC, abstractmethod

from pandas import DataFrame

from flox.learn.model import FloxModule


class Process(ABC):
    @abstractmethod
    def start(self, debug_mode: bool = False) -> tuple[FloxModule, DataFrame]:
        """Starts the FL process.

        Returns:
            The trained global module hosted on the leader of `topos`.
            The history metrics from training.
        """
