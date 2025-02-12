from __future__ import annotations

import abc
import typing as t
from collections import OrderedDict

import torch
from torch import nn

from flight.learning import AbstractModule
from flight.learning.params import Params
from flight.learning.types import FrameworkKind

from .types import TensorLoss, TensorStepOutput

_DEFAULT_INCLUDE_STATE: t.Final[bool] = False
"""
Default constant for whether to include the state in the parameters of a
[`TorchModule`][flight.learning.torch.module.TorchModule].
"""


class TorchModule(AbstractModule, nn.Module):
    """
    Wrapper class for a PyTorch model (i.e., `torch.nn.Module`).

    Based on PyTorch Lightning's [LightningModule](
        https://lightning.ai/docs/pytorch/stable/_modules/lightning/
        pytorch/core/module.html#LightningModule
    ).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_state = kwargs.get("include_state", _DEFAULT_INCLUDE_STATE)

    ####################################################################################

    # noinspection PyMethodMayBeStatic
    @t.final
    def kind(self) -> FrameworkKind:
        return "torch"

    def get_params(self) -> Params:
        """
        Getter method for the parameters of a trainable module (i.e., neural network)
        implemented in PyTorch.

        Returns:
            The parameters of the module. If the `to_numpy` flag is set to `True`,
                then `NpParams` are returned (i.e., values are NumPy `ndarray`s);
                `TorchParams` are returned (i.e., values are PyTorch `Tensor`s);

        Notes:
            We recommend not changing the `to_numpy` flag unless you are sure of what
            you are doing. The default value is set to `True` to allow for standard
            mathematical operations in aggregation functions across different
            frameworks.

        Throws:
            - `UnsupportedParameterKindError`: If the parameter kind is
                unknown/unsupported
        """
        if self.include_state:
            params = OrderedDict(
                (name, param) for name, param in self.state_dict().items()
            )
        else:
            params = OrderedDict(
                (name, param.data) for (name, param) in self.named_parameters()
            )

        return Params(params)

    def set_params(self, params: Params) -> None:
        """
        Setter method for the parameters of a trainable module (i.e., neural network)
        implemented in PyTorch.

        Args:
            params (Params): The parameters to set.

        Throws:
            - `ValueError`: if the parameter pair from (`next(iter(params.items())`)
                is not of length 2.
            - `Exception`: can be thrown if the parameters cannot be converted to a
                PyTorch `Tensor`s.
        """
        self.load_state_dict(params.torch(), strict=self.include_state, assign=False)

    ####################################################################################

    @abc.abstractmethod
    def training_step(self, *args: t.Any, **kwargs) -> TensorLoss:
        """
        Abstract method for performing a single training step.

        This method should be implemented in subclasses to define the training logic
        for the given module.

        Args:
            *args: Positional arguments that represent the input data or other relevant
                information required for training.
            **kwargs: Keyword arguments that represent additional settings or
                configurations for the training step.

        Returns:
            Loss returned from the training step. This is used for backpropagation and
                metric tracking.
        """

    @abc.abstractmethod
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Abstract method for configuring the optimizer(s) used during model training.

        This method should be implemented in subclasses to define the optimization
        strategy by returning a `torch.optim.Optimizer` instance or a list of
        optimizers. The optimizer manages the learning rate and other hyperparameters
        related to the model's weight updates during training.

        Returns:
            A configured optimizer or a list of optimizers for training the model.

        Throws:
            - `NotImplementedError`: If the method is not overridden in a subclass.
        """

    ####################################################################################

    def predict_step(self, *args: t.Any, **kwargs) -> TensorStepOutput:
        """
        Perform a single prediction step using the model.

        This method is responsible for making predictions on a batch of input data.
        The method returns the predictions in the form of a `LocalStepOutput` object,
        which typically contains the model's output for the given inputs.

        Args:
            *args (t.Any): Positional arguments that represent the input data or
                other relevant information required for prediction.
            **kwargs (t.Any): Keyword arguments that represent additional settings or
                configurations for the prediction step.

        Returns:
            The output of the prediction step, encapsulating the model's predictions.

        Throws:
            - `NotImplementedError`: If the method is not implemented.
        """
        raise NotImplementedError(
            "Method `predict_step` must be implemented if it is to be used."
        )

    def test_step(self, *args: t.Any, **kwargs) -> TensorStepOutput:
        """
        Perform a single testing step to evaluate the model's performance.

        Args:
            *args (t.Any): Positional arguments that represent the input data or
                other relevant information required for prediction.
            **kwargs (t.Any): Keyword arguments that represent additional settings or
                configurations for the prediction step.

        Returns:
            The output of the prediction step, encapsulating the model's predictions.

        Throws:
            - `NotImplementedError`: If the method is not implemented.
        """
        raise NotImplementedError(
            "Method `test_step` must be implemented if it is to be used."
        )

    def validation_step(self, *args: t.Any, **kwargs) -> TensorStepOutput:
        """
        Perform a single validation step to assess the model's performance on
        validation data.

        Args:
            *args (t.Any): Positional arguments that represent the input data or
                other relevant information required for prediction.
            **kwargs (t.Any): Keyword arguments that represent additional settings or
                configurations for the prediction step.

        Returns:
            The output of the prediction step, encapsulating the model's predictions.

        Throws:
            - `NotImplementedError`: If the method is not implemented.
        """
        raise NotImplementedError(
            "Method `validation_step` must be implemented if it is to be used."
        )
