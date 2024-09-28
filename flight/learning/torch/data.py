from __future__ import annotations

import abc
import typing as t

from torch.utils.data import DataLoader

from flight.learning import AbstractDataModule

if t.TYPE_CHECKING:
    from flight.federation.topologies import Node
    from flight.learning.types import DataKinds


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
