import typing as t

from torch import Tensor

LocalStepOutput: t.TypeAlias = t.Optional[Tensor | t.Mapping[str, t.Any]]
Params: t.TypeAlias = t.Mapping[str, Tensor]
Loss: t.TypeAlias = Tensor
