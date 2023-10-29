from torch.utils.data import Dataset, Subset
from typing import NewType, TypeVar, Union

from flox.flock import FlockNodeID


T_co = TypeVar("T_co", covariant=True)
FederatedSubsets = NewType(
    "FederatedSubsets", dict[FlockNodeID, Union[Dataset[T_co], Subset[T_co]]]
)
