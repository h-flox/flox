from __future__ import annotations

import typing as t

import numpy.typing as npt
from torch import Tensor
from torch.utils.data import Dataset

LocalStepOutput: t.TypeAlias = t.Optional[Tensor | t.Mapping[str, t.Any]]
"""
The output of a local training step, which can be a loss or a dictionary of...
"""

Params: t.TypeAlias = t.MutableMapping[str, Tensor | npt.NDArray]
"""
Type alias for model parameters; a mapping where the keys are strings and the
values are Tensors.
"""

Loss: t.TypeAlias = Tensor
"""
A type alias for the loss tensor.
"""

FrameworkKind = t.Literal["lightning", "scikit", "torch"]
"""
The deep learning framework used for a federation.
"""

DataKinds = t.Literal["train", "test", "validation"]
"""
The kinds of data that can be used for a federation.
"""

Data = t.Union[
    # npt.ArrayLike,
    npt.NDArray,
    Dataset,
]
