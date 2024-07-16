import typing as t

import torch

LocalStepOutput: t.TypeAlias = t.Optional[torch.Tensor | t.Mapping[str, t.Any]]
