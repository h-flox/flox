import abc
import typing as t

import torch
from torch import nn

LocalStepOutput: t.TypeAlias = t.Any


class RequiredMethodsMixin(abc.ABC):
    """Bye"""

    @abc.abstractmethod
    def train_step(self, *args: t.Any, **kwargs: t.Any) -> LocalStepOutput:
        """
        Hello

        Args:
            *args:
            **kwargs:

        Returns:

        """

    @abc.abstractmethod
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Helo

        Returns:

        """


class OptionalMethodsMixin(abc.ABC):
    """
    Hello
    """

    def validation_step(self, *args: t.Any, **kwargs: t.Any) -> LocalStepOutput:
        """
        Hello

        Args:
            *args:
            **kwargs:

        Returns:

        """


class FlightModule(abc.ABC, nn.Module, RequiredMethodsMixin, OptionalMethodsMixin):
    """
    Wrapper class for a PyTorch model (i.e., `torch.nn.Module`).

    Based on PyTorch Lightning's
    [LightningModule](https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/core/module.html#LightningModule).
    """
