from abc import ABC, abstractmethod

from pandas import DataFrame

from flox.data import FloxDataset, FederatedSubsets, LocalDataset
from flox.flock import FlockNode
from flox.nn.model import FloxModule


class BaseProcess(ABC):
    dataset: FloxDataset

    @abstractmethod
    def start(self) -> tuple[FloxModule, DataFrame]:
        """Starts the FL process.

        Returns:
            The trained global module hosted on the leader of `flock`.
            The history metrics from training.
        """

    def fetch_worker_data(self, node: FlockNode):
        match self.dataset:
            case FederatedSubsets():
                return self.dataset.load(node)
            case LocalDataset():
                return self.dataset.load(node)
            case _:
                cls_name = self.dataset.__class__.__name__
                raise TypeError(
                    f"`{cls_name}` is not a valid `FloxDataset`. Class member `dataset` must "
                    f"be a subclass of either `FederatedSubsets` or `LocalDataset`."
                )
