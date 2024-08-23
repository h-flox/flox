from __future__ import annotations

import abc
import typing as t

if t.TYPE_CHECKING:
    from ..federation.topologies import Node
    from ..types import Record
    from .types import Params

Data = t.TypeVar("Data")
DataKinds = t.Literal["train", "test", "validation"]


class AbstractDataModule(abc.ABC):
    """
    The standard abstraction for *data modules* in Flight.

    This class defines the necessary methods and attributes for data that can be used
    for training in federations in Flight. Flight provides implementations of this
    abstract class to provide support for training models using the following
    frameworks:

    - PyTorch, see [`TorchDataModule`][flight.learning.torch.TorchDataModule].
    - Scikit-Learn, see [`ScikitDataModule`][flight.learning.scikit.ScikitDataModule].
    """

    @abc.abstractmethod
    def train_data(
        self,
        node: Node | None = None,
    ) -> Data:
        """
        The **training data** returned by this data module.

        Args:
            node (Node | None): Node on which to load the data on.

        Returns:
            Data that will be used for training.
        """

    @abc.abstractmethod
    def test_data(self, node: Node | None = None) -> Data | None:
        """
        The **testing data** returned by this data module.

        Args:
            node (Node | None): Node on which to load the data on.

        Returns:
            Data that will be used for training.
        """

    @abc.abstractmethod
    def valid_data(self, node: Node | None = None) -> Data | None:
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


class AbstractModule(abc.ABC):
    @abc.abstractmethod
    def get_params(self) -> Params:
        """TODO"""
        pass

    @abc.abstractmethod
    def set_params(self, params: Params) -> None:
        """TODO"""
        pass


class AbstractTrainer(abc.ABC):
    def __init__(self, node: Node | None = None, **kwargs):
        self.node = node

    @abc.abstractmethod
    def fit(self, module: AbstractModule, data: AbstractDataModule) -> list[Record]:
        """TODO"""

    @abc.abstractmethod
    def test(self, module: AbstractModule, data: AbstractDataModule) -> list[Record]:
        """TODO"""

    @abc.abstractmethod
    def validate(
        self, module: AbstractModule, data: AbstractDataModule
    ) -> list[Record]:
        """TODO"""
