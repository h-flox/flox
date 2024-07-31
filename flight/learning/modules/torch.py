from __future__ import annotations

import abc
import typing as t
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader

if t.TYPE_CHECKING:
    from flight.federation.topologies import Node
    from flight.learning.types import LocalStepOutput, Params


_DEFAULT_INCLUDE_STATE = False


# class TorchTrainable:
#     def __init__(self, module: torch.nn.Module, include_state: bool = False) -> None:
#         self.module = module
#         self.include_state = include_state


class TorchDataModule(abc.ABC):
    # def __init__(self, *args, **kwargs):
    #     pass

    @abc.abstractmethod
    def train_data(self, node: Node | None = None) -> DataLoader:
        pass

    # noinspection PyMethodMayBeStatic
    def test_data(self, node: Node | None = None) -> DataLoader | None:
        return None

    # noinspection PyMethodMayBeStatic
    def valid_data(self, node: Node | None = None) -> DataLoader | None:
        return None


class FlightModule(nn.Module, abc.ABC):
    """
    Wrapper class for a PyTorch model (i.e., `torch.nn.Module`).

    Based on PyTorch Lightning's
    [LightningModule](https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/core/module.html#LightningModule).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_state = kwargs.get("include_state", _DEFAULT_INCLUDE_STATE)

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
        raise NotImplementedError()

    def test_step(self, *args: t.Any, **kwargs: t.Any) -> LocalStepOutput:
        """
        Hello

        Args:
            *args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError()

    def validation_step(self, *args: t.Any, **kwargs: t.Any) -> LocalStepOutput:
        """
        Hello

        Args:
            *args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError()

    def get_params(self) -> Params:
        params = self.module.state_dict()
        if not self.include_state:
            params = OrderedDict(
                [(name, value.data) for (name, value) in params if value.requires_grad]
            )

        return params

    def set_params(self, params: Params) -> None:
        if self.include_state:
            self.module.load_state_dict(
                params,
                strict=True,
                assign=False,
            )
        else:
            self.module.load_state_dict(
                params,
                strict=False,
                assign=False,
            )
