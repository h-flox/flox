from __future__ import annotations

import typing as t

from numpy import typing as npt
from torch import Tensor
from torch.utils.data import Dataset

LocalStepOutput: t.TypeAlias = t.Optional[Tensor | t.Mapping[str, t.Any]]
"""
The output of a local training step, which can be a loss or a dictionary of...
"""

TorchLocalStepOutput: t.TypeAlias = t.Optional[Tensor | t.Mapping[str, Tensor]]
"""
The output of a local training step, which can be a loss or a dictionary of...
"""

NpParams: t.TypeAlias = t.Dict[str, npt.NDArray]
"""
Type alias for model parameters as a mapping where the keys are strings and
the values are Numpy `ndarray`s.
"""

TorchParams: t.TypeAlias = t.Dict[str, Tensor]
"""
Type alias for model parameters as a mapping where the keys are strings and
the values are parameters as PyTorch `Tensor`s.
"""

Params: t.TypeAlias = NpParams | TorchParams
"""
Type alias for model parameters; a mapping where the keys are strings and the
values are parameters (as either Numpy `ndarray`s or PyTorch `Tensor`s).
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
FloatTriple: t.TypeAlias = tuple[float, float, float]
FloatDouble: t.TypeAlias = tuple[float, float]
DataIterable: t.TypeAlias = t.Iterable[Data] | npt.ArrayLike
