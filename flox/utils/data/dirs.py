from abc import abstractmethod
from torch.utils.data import Dataset

from flox.flock import FlockNodeID
from flox.utils.data.base import FederatedDataset, T_co


class FederatedDataDir(FederatedDataset):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load(self, idx: FlockNodeID) -> Dataset[T_co]:
        """
        ...

        Args:
            idx (FlockNodeID):

        Returns:
            Dataset[T_co]
        """
