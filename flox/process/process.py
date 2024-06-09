from abc import ABC, abstractmethod

from pandas import DataFrame

from flox.nn.model import FloxModule


class Process(ABC):
    @abstractmethod
    def start(self, debug_mode: bool = False) -> tuple[FloxModule, DataFrame]:
        """Starts the FL process.

        Returns:
            The trained global module hosted on the leader of `flock`.
            The history metrics from training.
        """
