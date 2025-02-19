from __future__ import annotations

import typing as t

import torch

TensorLoss: t.TypeAlias = torch.Tensor
"""
Single loss tensor type alias.
"""

TensorStepOutput: t.TypeAlias = t.Optional[torch.Tensor | t.Mapping[str, torch.Tensor]]
"""
Step tensor type alias.
"""
