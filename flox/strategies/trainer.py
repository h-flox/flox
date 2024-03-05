import typing as t

import torch

Loss: t.TypeAlias = torch.Tensor

if t.TYPE_CHECKING:
    pass


class TrainerStrategy(t.Protocol):
    def before_backprop(self, loss: Loss) -> Loss:
        pass

    def after_backprop(self, loss: Loss) -> Loss:
        pass
