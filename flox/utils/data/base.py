from abc import ABC
from pathlib import Path
from torch.utils.data import Dataset, Subset
from typing import TypeVar

from flox.flock import FlockNodeID

T_co = TypeVar("T_co", covariant=True)


class FederatedData(ABC):
    def __init__(self):
        pass

    def load(self, node: FlockNodeID) -> Dataset[T_co]:
        pass


class FederatedDataSubsets(FederatedData):
    def __init__(self):
        super().__init__()

    def load(self, node: FlockNodeID) -> Subset[T_co]:
        pass


class FederatedDataDir(FederatedData):
    def __init__(self, data_dir: Path):
        super().__init__()
        self.data_dir = data_dir

    def load(self, node: FlockNodeID) -> Dataset[T_co]:
        pass


if __name__ == "__main__":

    class MyData(FederatedDataDir):
        def load(self, name: FlockNodeID) -> Dataset[T_co]:
            pass
