from enum import IntEnum, auto
from typing import NewType, TypeVar, Union

from torch.utils.data import Dataset, Subset

from flox.flock import FlockNodeID
from flox.flock.states import NodeState


class FloxDatasetKind(IntEnum):
    STANDARD = auto()
    FEDERATED = auto()
    INVALID = auto()

    @staticmethod
    def from_obj(obj) -> "FloxDatasetKind":
        if isinstance(obj, Dataset):
            return FloxDatasetKind.STANDARD
        elif isinstance(obj, FederatedSubsets):
            return FloxDatasetKind.FEDERATED
        else:
            return FloxDatasetKind.INVALID


def flox_compatible_data(obj) -> bool:
    kind = FloxDatasetKind.from_obj(obj)
    if kind is FloxDatasetKind.INVALID:
        return False
    return True


T_co = TypeVar("T_co", covariant=True)
FederatedSubsets = NewType(
    "FederatedSubsets", dict[FlockNodeID, Union[Dataset[T_co], Subset[T_co]]]
)


class MyFloxDataset(Dataset):
    def __init__(self, state: NodeState, /, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state


FloxDataset = NewType("FloxDataset", Union[MyFloxDataset, FederatedSubsets])
