from __future__ import annotations

import abc
import typing as t
from collections import OrderedDict

import torch
from torch import nn

from .base import AbstractDataModule, AbstractModule, DataKinds
from .types import LocalStepOutput, Params

if t.TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from ..federation.topologies import Node

_DEFAULT_INCLUDE_STATE = False
"""
...
"""


class TorchDataModule(AbstractDataModule):
    """
    An abstract class meant for data objects compatible with PyTorch (i.e., PyTorch
    `Dataset` and `DataLoader` objects).

    When using PyTorch in Flight, this class should be extended by any custom class
    used to load in your own datasets, pre-process/transform them accordingly, and
    prepare them to return as `DataLoader`s.

    This class does not do much to handle the loading of data into Flight. It simply
    provides the abstract methods that need to be overridden by users to define their
    own data modules. It also requires that type interface, specifically that each
    data loading method (i.e., `train_data()`, `test_data()`, and `valid_data()`)
    returns a `DataLoader`.

    Node-specific logic for loading in data (either from disc or from memory) must be
    provided by an implementation of this class.

    An example of how this class would be used can be found below.

    Examples:
        >>> import torch
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>>
        >>> class MyTorchDataModule(TorchDataModule):
        >>>     '''Sample data module that only provides training data.'''
        >>>     def __init__(
        >>>         self,
        >>>         sizes: list[int],
        >>>         seeds: list[int],
        >>>         batch_size: int = 32
        >>>     ) -> None:
        >>>         self.sizes = sizes  # The number of samples per node (by index).
        >>>         self.seeds = seeds  # The seeds per node (by index).
        >>>         self.batch_size = batch_size
        >>>
        >>>     def generate_data(self, i: int) -> TensorDataset:
        >>>         '''Helper function for generating data with `randn` and seed.'''
        >>>         g = torch.Generator(device="cpu")
        >>>         g.manual_seed(self.seeds[i])
        >>>         tensors = torch.randn((self.sizes[i], 1))
        >>>         return TensorDataset(tensors)
        >>>
        >>>     def train_data(self, node = None) -> DataLoader:
        >>>         assert node is not None
        >>>         bs = self.batch_size
        >>>         return DataLoader(self.generate_data(node.idx), batch_size=bs)
        >>>
        >>>     def size(self, node = None, kind = "train"):
        >>>         assert node is not None and kind == "train"
        >>>         return self.sizes[node.idx]
        >>>
        >>>     ...
    """

    @abc.abstractmethod
    def train_data(self, node: Node | None = None) -> DataLoader:
        """
        The **training data** returned by this data module.

        Args:
            node (Node | None): Node on which to load the data on.

        Returns:
            Data that will be used for training.
        """

    @abc.abstractmethod
    def test_data(self, node: Node | None = None) -> DataLoader | None:
        """
        The **testing data** returned by this data module.

        Args:
            node (Node | None): Node on which to load the data on.

        Returns:
            Data that will be used for training.
        """

    @abc.abstractmethod
    def valid_data(self, node: Node | None = None) -> DataLoader | None:
        """
        The **validation data** returned by this data module.

        Args:
            node (Node | None): Node on which to load the data on.

        Returns:
            Data that will be used for validation.
        """

    # noinspection PyMethodMayBeStatic
    def size(self, node: Node | None = None, kind: DataKinds = "train") -> int | None:
        """
        If implemented, this should return the size of the dataset.

        Args:
            node (Node | None): Node on which to load the data on.
            kind (DataKinds): The kind of data to get the size of
                (namely, `'train'`, `'test'`, or `'validation'`).

        Returns:
            The size of the respective dataset.
        """
        return None


class TorchModule(AbstractModule, nn.Module):
    """
    Wrapper class for a PyTorch model (i.e., `torch.nn.Module`).

    Based on PyTorch Lightning's
    [LightningModule](
        https://lightning.ai/docs/pytorch/stable/_modules/lightning/
        pytorch/core/module.html#LightningModule
    ).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_state = kwargs.get("include_state", _DEFAULT_INCLUDE_STATE)

    ####################################################################################

    def get_params(self) -> Params:
        state_dict = self.state_dict()
        if self.include_state:
            return state_dict
        else:
            param_names = dict(self.named_parameters())
            return OrderedDict(
                [
                    (name, value.data)
                    for (name, value) in state_dict.items()
                    if name in param_names
                ]
            )

    def set_params(self, params: Params) -> None:
        if self.include_state:
            self.load_state_dict(
                params,
                strict=True,
                assign=False,
            )
        else:
            self.load_state_dict(
                params,
                strict=False,
                assign=False,
            )

    ####################################################################################

    @abc.abstractmethod
    def training_step(self, *args: t.Any, **kwargs) -> LocalStepOutput:
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
        Abstract method for configuring the optimizer(s) used during model training.

        This method should be implemented in subclasses to define the optimization
        strategy by returning a `torch.optim.Optimizer` instance or a list of
        optimizers. The optimizer manages the learning rate and other hyperparameters
        related to the model's weight updates during training.

        Returns:
            A configured optimizer or a list of optimizers for training the model.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """

    ####################################################################################

    def predict_step(self, *args: t.Any, **kwargs) -> LocalStepOutput:
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

        Raises:
            - `NotImplementedError`: If the method is not implemented.
        """
        # TODO: Revise pydoctstrings.
        raise NotImplementedError()

    def test_step(self, *args: t.Any, **kwargs) -> LocalStepOutput:
        """
        Perform a single testing step to evaluate the model's performance.

        Args:
            *args (t.Any): Positional arguments that represent the input data or
                other relevant information required for prediction.
            **kwargs (t.Any): Keyword arguments that represent additional settings or
                configurations for the prediction step.

        Returns:
            The output of the prediction step, encapsulating the model's predictions.

        Raises:
            - `NotImplementedError`: If the method is not implemented.
        """
        raise NotImplementedError()

    def validation_step(self, *args: t.Any, **kwargs) -> LocalStepOutput:
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

        Raises:
            - `NotImplementedError`: If the method is not implemented.
        """
        raise NotImplementedError()
