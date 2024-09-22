from __future__ import annotations

import abc
import typing as t
from collections import OrderedDict

import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from ..federation.topologies.node import WorkerState
from ..strategies import TrainerStrategy
from ..types import Record
from .base import AbstractDataModule, AbstractModule
from .types import DataKinds, FrameworkKind, LocalStepOutput, Params

if t.TYPE_CHECKING:
    from ..federation.topologies import Node

    # TODO: Replace this accordingly.
    EVAL_DATALOADERS: t.TypeAlias = t.Any  #
    TRAIN_DATALOADERS: t.TypeAlias = t.Any  #
    _PATH: t.TypeAlias = t.Any  #
    LightningDataModule: t.TypeAlias = t.Any  #

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

    def test_data(self, node: Node | None = None) -> DataLoader | None:
        """
        The **testing data** returned by this data module.

        Args:
            node (Node | None): Node on which to load the data on.

        Returns:
            Data that will be used for training.
        """
        return None

    def valid_data(self, node: Node | None = None) -> DataLoader | None:
        """
        The **validation data** returned by this data module.

        Args:
            node (Node | None): Node on which to load the data on.

        Returns:
            Data that will be used for validation.
        """
        return None

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

    # noinspection PyMethodMayBeStatic
    @t.final
    def kind(self) -> FrameworkKind:
        return "torch"

    def get_params(self, to_numpy: bool = True) -> Params:
        """
        Getter method for the parameters of a trainable module (i.e., neural network)
        implemented in PyTorch.

        Args:
            to_numpy (bool): Flag to convert the parameters to numpy arrays. Defaults
                to `True`.

        Returns:
            The parameters of the module.

        Notes:
            We recommend not changing the `to_numpy` flag unless you are sure of what
            you are doing. The default value is set to `True` to allow for standard
            mathematical operations in aggregation functions across different
            frameworks.
        """

        def _parse_params(pair: tuple[str, torch.Tensor]):
            """
            Helper hidden function that converts parameters to NumPy `ndarray`s if
            specified by the `get_params` arg.
            """
            if to_numpy:
                return pair[0], pair[1].data.numpy()
            else:
                return pair[0], pair[1].data

        state_dict = self.state_dict()
        if self.include_state:
            return OrderedDict(_parse_params(items) for items in state_dict.items())
        else:
            param_names = dict(self.named_parameters())
            return OrderedDict(
                _parse_params((name, value))
                for (name, value) in state_dict.items()
                if name in param_names
            )

    def set_params(self, params: Params) -> None:
        """
        Setter method for the parameters of a trainable module (i.e., neural network)
        implemented in PyTorch.

        Args:
            params (Params): The parameters to set.

        Throws:
            An `Exception` can be thrown. if the parameter cannot be converted to a
                PyTorch `Tensor`.
        """

        def _parse_params(pair: tuple[str, torch.Tensor]):
            """
            Helper hidden function that converts parameters to PyTorch `Tensor`s if
            specified by the `get_params` arg.
            """
            if isinstance(pair[1], torch.Tensor):
                return pair[0], pair[1]
            try:
                return pair[0], torch.tensor(pair[1])
            except Exception as err:
                err.add_note("Failed to convert parameter to PyTorch `Tensor`.")
                raise err

        strict = self.include_state
        new_params = OrderedDict(_parse_params(items) for items in params.items())
        return self.load_state_dict(new_params, strict=strict, assign=False)

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


class TorchTrainer:
    def __init__(
        self,
        max_epochs: int = 1,
        node: Node | None = None,
        strategy: TrainerStrategy | None = None,
        progress_bar: bool = True,
    ):
        self.node = node
        self.strategy = strategy
        self.max_epochs = max_epochs
        self._progress_bar = progress_bar
        self._curr_step = 0
        self._results = []

        if self.node is None:
            from flight.federation.topologies import Node

            self.node = Node(idx=0, kind="worker")

        if self.strategy is None:
            from flight.strategies.base import DefaultTrainerStrategy

            self.strategy = DefaultTrainerStrategy()

        # self._logger =

        try:
            self._device = torch.device(node.extra.get("device", "cpu"))
        except (AttributeError, TypeError):
            self._device = torch.device("cpu")

    def fit(
        self,
        node_state: WorkerState,
        model: TorchModule,
        data: TorchDataModule,
        validate_every_n_epochs: int = 1,
        ckpt_path: _PATH | None = None,
    ) -> list[Record]:
        """

        Args:
            node_state (WorkerState):
            model (TorchModule):
            data (TorchDataModule):
            validate_every_n_epochs:
            ckpt_path:

        Raises:
            - ValueError: Thrown when illegal values are given to arguments.

        Returns:

        """
        # TODO: Run the argument validation in a separate utility function to keep
        #  this function light.
        if validate_every_n_epochs < 1:
            raise ValueError("Illegal value for argument `validate_every_n_epochs`.")

        model.to(self._device)

        train_dataloader = data.train_data(self.node)
        valid_dataloader = data.valid_data(self.node)

        if not isinstance(train_dataloader, DataLoader):
            raise TypeError(
                "Method for argument `data.train_data(.)` must return a `DataLoader`."
            )
        if not isinstance(valid_dataloader, DataLoader | None):
            raise TypeError(
                "Method for argument `data.valid_data(.)` must return a `DataLoader` "
                "or `None`."
            )

        pbar_prefix = f"TorchTrainer(NodeID={self.node.idx})"
        if self._progress_bar:
            total = len(train_dataloader)
            if valid_dataloader is not None:
                total += len(valid_dataloader)
            total *= self.max_epochs
            pbar = tqdm.tqdm(total=total, desc=pbar_prefix)
        else:
            pbar = None

        optimizer = model.configure_optimizers()

        self._curr_step = 0
        self._results = []  # TODO: Convert to logger.

        for epoch in range(self.max_epochs):
            if pbar:
                pbar.set_description(f"{pbar_prefix} | {epoch=}")
            train_losses = self._epoch(
                epoch,
                node_state,
                model,
                optimizer,
                train_dataloader,
                pbar,
            )
            for loss in train_losses:
                self._results.append({"epoch": epoch, "train/loss": loss.item()})

            to_validate = all(
                [epoch % validate_every_n_epochs == 0, valid_dataloader is not None]
            )
            if to_validate:
                val_losses = self.validate(epoch, model, valid_dataloader, pbar)
                for loss in val_losses:
                    self._results.append(
                        {
                            "epoch": epoch,
                            "val/loss": loss.item(),
                            "step": self._curr_step,
                        }
                    )

        return self._results

    def _epoch(
        self,
        epoch: int,
        node_state: WorkerState,
        model: TorchModule,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        pbar: tqdm.tqdm | None,
    ):
        self._set_train_mode(model, True)

        losses = []
        for batch_idx, batch in enumerate(dataloader):
            batch = self._batch_to_device(batch)
            loss = model.training_step(batch, batch_idx)

            # Perform backpropagation and call trainer strategy callbacks.
            optimizer.zero_grad()
            if self.strategy is not None:
                loss = self.strategy.before_backprop(node_state, loss)
            loss.backward()
            if self.strategy is not None:
                loss = self.strategy.after_backprop(node_state, loss)
            optimizer.step()

            self._results.append(
                {
                    "epoch": epoch,
                    "train/loss": loss.item(),
                    "train/batch_idx": batch_idx,
                    "train/step": self._curr_step,
                }
            )
            self._curr_step += 1

            if pbar is not None:
                pbar.update()

        return losses

    def validate(
        self,
        epoch: int,
        model: TorchModule,
        dataloader: DataLoader,
        pbar: tqdm.tqdm | None,
        *args,
        **kwargs,
    ):
        self._set_train_mode(model, False)

        losses = []
        for batch_idx, batch in enumerate(dataloader):
            batch = self._batch_to_device(batch)
            loss = model.validation_step(batch, batch_idx)
            if loss is not None:
                self._results.append(
                    {
                        "epoch": epoch,
                        "valid/loss": loss.item(),
                        "valid/batch_idx": batch_idx,
                        "valid/step": self._curr_step,
                    }
                )
            else:
                pass

            if pbar is not None:
                pbar.update()

        return losses

    def _batch_to_device(self, batch: tuple[t.Any, ...]):
        items = []
        for item in batch:
            try:
                item = item.to(self._device)
            except AttributeError:
                pass
            items.append(item)
        return tuple(items)

    @staticmethod
    def _set_train_mode(model: TorchModule, train_mode: bool = True) -> None:
        """
        Hidden utility function that switches the `TorchTrainer` to training or
        validation mode.

        Args:
            model (TorchModule): Model to set to training or evaluation mode.
            train_mode (bool): Training mode flag.
        """
        torch.set_grad_enabled(train_mode)
        if train_mode:
            model.train()
        else:
            model.eval()
