import abc
import typing as t

import torch
from torch import nn

if t.TYPE_CHECKING:
    from .types import LocalStepOutput


class FlightModule(nn.Module, abc.ABC):
    """
    Wrapper class for a PyTorch model (i.e., `torch.nn.Module`).

    Based on PyTorch Lightning's
    [LightningModule](https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/core/module.html#LightningModule).
    """

    @abc.abstractmethod
    def training_step(self, *args: t.Any, **kwargs: t.Any) -> LocalStepOutput:
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

    def predict_step(self, *args: t.Any, **kwargs: t.Any) -> LocalStepOutput:
        """
        Hello

        Args:
            *args:
            **kwargs:

        Returns:

        """

    def test_step(self, *args: t.Any, **kwargs: t.Any) -> LocalStepOutput:
        """
        Hello

        Args:
            *args:
            **kwargs:

        Returns:

        """

    def validation_step(self, *args: t.Any, **kwargs: t.Any) -> LocalStepOutput:
        """
        Hello

        Args:
            *args:
            **kwargs:

        Returns:

        """
