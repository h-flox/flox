from __future__ import annotations

from torch.utils.data import Dataset
from typing import NewType, Union

from flox.data.subsets import FederatedSubsets

FloxDataset = NewType("FloxDataset", Union[Dataset, FederatedSubsets])
