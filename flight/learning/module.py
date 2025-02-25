from __future__ import annotations

import abc
import typing as t

import torch.nn as nn

if t.TYPE_CHECKING:
    from torch.optim import Optimizer


class TorchModule(abc.ABC, nn.Module):

    @abc.abstractmethod
    def configure_criterion(self, *args, **kwargs) -> t.Callable:
        """
        Configures the criterion (i.e., loss function) used for training the model.

        Returns:
            The criterion object to be used for training
        """

    @abc.abstractmethod
    def configure_optimizers(self, *args, **kwargs) -> Optimizer:
        """
        Configures the optimizers used for training the model.

        Returns:
            The optimizer object to be used for training.
        """


class TorchDataModule:
    def train_data(self, *args, **kwargs):
        pass

    def valid_data(self, *args, **kwargs):
        pass

    def test_data(self, *args, **kwargs):
        pass
