from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    import torch

    Loss: t.TypeAlias = torch.Tensor


class TrainerStrategy(t.Protocol):
    def before_backprop(self, state, loss: Loss) -> Loss:
        pass

    def after_backprop(self, state, loss: Loss) -> Loss:
        pass
