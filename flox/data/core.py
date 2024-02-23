from collections.abc import Mapping
from enum import IntEnum, auto
from typing import NewType, Union, get_args

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
        elif FloxDatasetKind.is_federated_dataset(obj):
            return FloxDatasetKind.FEDERATED
        else:
            return FloxDatasetKind.INVALID

    @staticmethod
    def is_federated_dataset(obj) -> bool:
        if not isinstance(obj, Mapping):
            return False

        return all(
            isinstance(k, get_args(FlockNodeID)) and isinstance(v, (Dataset, Subset))
            for k, v in obj.items()
        )


def flox_compatible_data(obj) -> bool:
    kind = FloxDatasetKind.from_obj(obj)
    if kind is FloxDatasetKind.INVALID:
        return False
    return True


FederatedSubsets = NewType(
    "FederatedSubsets", Mapping[FlockNodeID, Union[Dataset, Subset]]
)


class MyFloxDataset(Dataset):
    def __init__(self, state: NodeState, /, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state


FloxDataset = Union[MyFloxDataset, FederatedSubsets]
