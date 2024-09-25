from __future__ import annotations

import abc
import typing as t

if t.TYPE_CHECKING:
    from ..federation.topologies import Node
    from ..types import Record
    from .types import Data, DataKinds, FrameworkKind, Params


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

    _KIND: t.Final[str] = "abstract"

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

    def test_data(self, node: Node | None = None) -> Data | None:
        """
        The **testing data** returned by this data module.

        Args:
            node (Node | None): Node on which to load the data on.

        Returns:
            Data that will be used for training.
        """
        return None

    def valid_data(self, node: Node | None = None) -> Data | None:
        """
        The **validation data** returned by this data module.

        Args:
            node (Node | None): Node on which to load the data on.

        Returns:
            Data that will be used for validation.
        """
        return None

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
        """
        Getter method for the parameters of a trainable module (i.e., neural network).

        Returns:
            The parameters of the module.
        """

    @abc.abstractmethod
    def set_params(self, params: Params) -> None:
        """
        Setter method for the parameters of a trainable module (i.e., neural network).

        Args:
            params (Params): The parameters to set.
        """

    @abc.abstractmethod
    def kind(self) -> FrameworkKind:
        """
        Getter method for the kind of framework this module is based on.

        This is an attribute that will be set by any subclass of this abstract class.

        Returns:
            The kind of framework this module is based on.
        """


class AbstractTrainer(abc.ABC):
    def __init__(self, node: Node | None = None, **kwargs):
        self.node = node

    @abc.abstractmethod
    def fit(self, module: AbstractModule, data: AbstractDataModule) -> list[Record]:
        """
        Fits (or trains) a module on a given data module.

        Args:
            module (AbstractModule): The module to fit.
            data (AbstractDataModule): The data module to use for training.

        Returns:
            A list of records containing the results of the training.
        """

    # @t.overload
    # def test(self, module: TorchModule, data: TorchDataModule) -> list[Record]:
    #     pass
    #
    # @t.overload
    # def test(self, module: ScikitModule, data: ScikitDataModule) -> list[Record]:
    #     pass

    @abc.abstractmethod
    def test(self, module: AbstractModule, data: AbstractDataModule) -> list[Record]:
        """
        Tests a module on a given data module by loading its testing data.

        Args:
            module (AbstractModule): The module to test
            data (AbstractDataModule): The data module to use for testing.

        Returns:
            A list of records containing the results of the testing.
        """

    @abc.abstractmethod
    def validate(
        self, module: AbstractModule, data: AbstractDataModule
    ) -> list[Record]:
        """
        Validates (or evaluates) a module on a given data module by loading its
        validation data.

        Args:
            module (AbstractModule): The module to validate.
            data (AbstractDataModule): The data module to use for validation.

        Returns:
            A list of records containing the results of the validation.
        """
