from __future__ import annotations

import abc
import collections as c
import enum
import functools
import typing as t

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from flight.system.topology import Node, NodeID

if t.TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader, Dataset


_DEFAULT_INCLUDE_STATE: t.Final[bool] = False
"""
Default constant for whether to include the state in the parameters of a
[`TorchModule`][flight.learning.torch.module.TorchModule].
"""

NpParams: t.TypeAlias = dict[str, np.ndarray]
"""
Type alias for model parameters as a mapping where the keys are strings and
the values are Numpy `ndarray`s.
"""

TorchParams: t.TypeAlias = dict[str, torch.Tensor]
"""
Type alias for model parameters as a mapping where the keys are strings and
the values are parameters as PyTorch `Tensor`s.
"""


class UnsupportedParameterKindError(ValueError):
    """
    An Exception raised when an unsupported parameter kind is detected.
    """

    def __init__(self, message: str | None = None, *args):
        if message is None:
            message = (
                "The parameter kind is unknown or unsupported. "
                "Please refer to the docs."
            )
        super().__init__(message, *args)


class InconsistentParamValuesError(ValueError):
    """
    An Exception raised when the parameter value kinds are inconsistent.
    """

    def __init__(self, message: str | None = None, *args):
        if message is None:
            message = "The parameter values are inconsistent. Please refer to the docs."
        super().__init__(message, *args)


class ParamKinds(enum.Enum):
    """
    An enumeration of the kinds of parameters supported by Flight.
    """

    NUMPY = enum.auto()
    """
    Parameters implemented as NumPy `ndarray`s.
    """

    TORCH = enum.auto()
    """
    Parameters implemented as PyTorch `Tensor`s.
    """


def infer_param_kind(param: np.ndarray | torch.Tensor) -> ParamKinds:
    """
    Detect the kind of parameter.

    Args:
        param (np.ndarray | torch.Tensor): The parameter to infer the type for.

    Returns:
        The kind of parameter.

    Throws:
        - `UnsupportedParameterKindError`: If the parameter kind is unknown/unsupported.
    """
    if isinstance(param, np.ndarray):
        return ParamKinds.NUMPY
    elif isinstance(param, torch.Tensor):
        return ParamKinds.TORCH
    else:
        raise UnsupportedParameterKindError()


def validate_param_kind(params: dict[str, t.Any]) -> ParamKinds:
    """
    Validate the kind of parameters.

    This function returns the kind of parameters (similar to `infer_param_kind`), but
    it will throw an error in the case where the parameters are not of the same kind.

    Args:
        params (dict[str, t.Any]): Parameters to infer the kind of.

    Returns:
        The kind of parameters if they are of the same kind. Otherwise, an error is
        thrown.

    Throws:
        - `InconsistentParamValuesError`: If the parameter values are inconsistent.
        - `UnsupportedParameterKindError`: If the parameter kind is unknown/unsupported.
            This will be thrown by the `infer_param_kind` function.
    """
    param_kinds = set(map(infer_param_kind, params.values()))
    if len(param_kinds) != 1:
        raise InconsistentParamValuesError()
    return param_kinds.pop()


class Params(c.OrderedDict):
    """
    A wrapper class for model parameters, implemented as an `OrderedDict`.

    Throws:
        - `InconsistentParamValuesError`: If the parameter values are inconsistent.
        - `UnsupportedParameterKindError`: If the parameter kind is unknown/unsupported.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def numpy(self) -> NpParams:
        """
        Convert the parameters to NumPy `ndarray`s.

        Returns:
            The parameters in NumPy `ndarray`s.
        """
        match self.inferred_kind:
            case ParamKinds.NUMPY:
                return self
            case ParamKinds.TORCH:
                return Params((k, v.numpy()) for k, v in self.items())

    def torch(self) -> TorchParams:
        """
        Convert the parameters to PyTorch `Tensor`s.

        Returns:
            The parameters in the PyTorch `Tensor`s.
        """
        match self.inferred_kind:
            case ParamKinds.TORCH:
                return self
            case ParamKinds.NUMPY:
                return Params((k, torch.from_numpy(v)) for k, v in self.items())

    @functools.cached_property
    def inferred_kind(self) -> ParamKinds:
        """
        The inferred kind of the parameters.

        Returns:
            The kind of parameters.
        """
        return validate_param_kind(self)


class TorchModule(abc.ABC, nn.Module):
    """
    A lightweight wrapper class for using a PyTorch model
    (i.e., `torch.nn.Module`) in Flight.

    Based on PyTorch Lightning's [LightningModule](
        https://lightning.ai/docs/pytorch/stable/_modules/lightning/
        pytorch/core/module.html#LightningModule
    ).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_state = kwargs.get("include_state", _DEFAULT_INCLUDE_STATE)

    @abc.abstractmethod
    def configure_criterion(self, *args, **kwargs) -> t.Callable:
        """
        Configures the criterion (i.e., loss function) used for training the model.

        Returns:
            The criterion object to be used for training.
        """

    @abc.abstractmethod
    def configure_optimizers(self, *args, **kwargs) -> Optimizer:
        """
        Configures the optimizers used for training the model.

        Returns:
            The optimizer object to be used for training.
        """

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
            params = c.OrderedDict(
                (name, param) for name, param in self.state_dict().items()
            )
        else:
            params = c.OrderedDict(
                (name, param.data) for name, param in self.named_parameters()
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
        self.load_state_dict(
            params.torch(),
            strict=self.include_state,
            assign=False,
        )


# TODO: We need to adjust the 'Examples' bit of the pydocstring for this class.
class TorchDataModule:
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
    def train_data(self, *args, **kwargs) -> DataLoader:
        """
        The **training data** returned by this data module.

        Args:
            node (Node | None): Node on which to load the data on.

        Returns:
            Data that will be used for training.
        """

    def test_data(self, *args, **kwargs) -> DataLoader | None:
        """
        The **testing data** returned by this data module.

        Args:
            node (Node | None): Node on which to load the data on.

        Returns:
            Data that will be used for training.
        """
        return None

    def valid_data(self, *args, **kwargs) -> DataLoader | None:
        """
        The **validation data** returned by this data module.

        Args:
            node (Node | None): Node on which to load the data on.

        Returns:
            Data that will be used for validation.
        """
        return None

    def size(self, kind: str = "train", /, **kwargs) -> int | None:
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


class FederatedDataModule(TorchDataModule):
    """
    This class defines a DataModule that is split across worker nodes in a federation's
    topology.

    This is especially helpful for simulation-based federations that are run with
    Flight. Rather than needing to manually define the logic to load data that are
    sharded across workers in a federation, this class simply requires the original
    dataset and the indices for training, testing, and validation data for each
    worker.

    A good analogy for this class is to think of it as the federated version of
    PyTorch's [`Subset`](https://pytorch.org/docs/stable/data.html#
    torch.utils.data.Subset) class.

    Notes:
        ==Currently, only supports PyTorch `Dataset` objects.==

    """

    def __init__(
        self,
        dataset: Dataset,
        train_indices: t.Mapping[NodeID, t.Sequence[int]],
        test_indices: t.Mapping[NodeID, t.Sequence[int]] | None,
        valid_indices: t.Mapping[NodeID, t.Sequence[int]] | None,
        **dataloader_kwargs,
    ) -> None:
        """
        Args:
            dataset (Dataset): The dataset to split across workers.
            train_indices (t.Mapping[NodeID, t.Sequence[int]]): The indices for the
                training data for each worker.
            test_indices (t.Mapping[NodeID, t.Sequence[int]] | None): The indices
                for the test data for each worker.
            valid_indices (t.Mapping[NodeID, t.Sequence[int]] | None): The indices
                for the validation data for each worker.
            **dataloader_kwargs: Keyword arguments to pass to the `DataLoader` class
                when calling `train_data()`, `valid_data()`, and `test_data()`.

        Throws:
            - `NotImplementedError`: If the dataset is not mappable.

        """
        self.dataset = dataset
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.valid_indices = valid_indices
        self.dataloader_kwargs = dataloader_kwargs

        try:
            self.dataset[0]
        except IndexError:
            pass
        except NotImplementedError as err:
            err.add_note(
                "Your PyTorch `Dataset` must be mappable (i.e., implements "
                "`__getitem__`)."
            )
            raise err

    def __iter__(self) -> t.Iterator[tuple[NodeID, DataLoader]]:
        """
        Returns an iterator over the workers and their respective datasets.

        Returns:
            An iterator over the workers and their respective datasets.
        """
        for worker_idx in self.train_indices:
            yield worker_idx, self.train_data(worker_idx)

    def __len__(self) -> int:
        """
        Returns the number of workers that have shards of the original dataset.

        Returns:
            Number of workers.
        """
        return len(self.train_indices)

    def __contains__(self, node_or_idx: Node | NodeID) -> bool:
        """
        Checks if a (worker) node exists in the federated data module.

        Args:
            node_or_idx (Node | NodeID): The node or node index to check exists in the
                federated data module.

        Returns:
            `True` if the node exists in the federated data module; `False` otherwise.
        """
        if isinstance(node_or_idx, Node):
            node_idx = node_or_idx.idx
        elif isinstance(node_or_idx, NodeID):
            node_idx = node_or_idx
        else:
            raise ValueError(
                f"FedDataModule.__contains__ only accepts arguments of type "
                f"`Node` or `NodeID`; got `{type(node_or_idx)}`."
            )

        return node_idx in self.train_indices

    def train_data(self, node: Node | NodeID | None = None) -> DataLoader:
        return self._get_data(node, self.train_indices)

    def test_data(self, node: Node | NodeID | None = None) -> DataLoader | None:
        if self.test_indices is not None:
            return self._get_data(node, self.test_indices)
        return None

    def valid_data(self, node: Node | None = None) -> DataLoader | None:
        if self.valid_indices is not None:
            return self._get_data(node, self.valid_indices)
        return None

    def _get_data(
        self,
        node: Node | NodeID | None,
        indices: t.Mapping[NodeID, t.Sequence[int]],
    ) -> DataLoader:
        from torch.utils.data import DataLoader

        node_id = self._resolve_node(node)
        subset = Subset(self.dataset, indices[node_id])
        return DataLoader(subset, **self.dataloader_kwargs)

    def _resolve_node(self, node_or_idx: Node | NodeID | None = None) -> NodeID:
        if node_or_idx is None:
            raise ValueError(
                "`node` argument for {} cannot be `None`.".format(
                    self.__class__.__name__
                )
            )
        elif isinstance(node_or_idx, Node):
            node_id = node_or_idx.idx
        elif isinstance(node_or_idx, NodeID):
            node_id = node_or_idx
        else:
            try:
                node_id = int(node_or_idx)
            except Exception as e:
                e.add_note(
                    f"`node` must be an instance of `Node` or `NodeID`, "
                    f"got {type(node_or_idx)}."
                )
                raise e

        return node_id
