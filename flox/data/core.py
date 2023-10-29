from enum import auto, IntEnum
from torch.utils.data import Dataset

from flox.data.subsets import FederatedSubsets


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
    pass
