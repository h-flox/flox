from __future__ import annotations

import abc
import typing as t
from collections import OrderedDict

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings  # noqa

from .base import AbstractDataModule, AbstractModule, AbstractTrainer

if t.TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    from ..federation.topologies import Node
    from ..types import Record
    from .types import FrameworkKind, Params


class ScikitDataModule(AbstractDataModule):
    WEIGHT_KEY_PREFIX: t.Final[str] = "weight"
    BIAS_KEY_PREFIX: t.Final[str] = "bias"

    @abc.abstractmethod
    def train_data(self, node: Node | None = None) -> ArrayLike:
        """
        The **training data** returned by this data module.

        Args:
            node (Node | None): Node on which to load the data on. Defaults to `None`.

        Returns:
            Data that will be used for training.
        """

    @abc.abstractmethod
    def test_data(self, node: Node | None = None) -> ArrayLike | None:
        """
        The **testing data** returned by this data module.

        Args:
            node (Node | None): Node on which to load the data on. Defaults to `None`.

        Returns:
            Data that will be used for testing. If `None`, then no testing is done.
        """

    @abc.abstractmethod
    def valid_data(self, node: Node | None = None) -> ArrayLike | None:
        """
        The **training data** returned by this data module.

        Args:
            node (Node | None): Node on which to load the data on. Defaults to `None`.

        Returns:
            Data that will be used for training. If `None`, then no validation is done.
        """


class ScikitModule(AbstractModule):
    WEIGHT_KEY_PREFIX = "weight"
    BIAS_KEY_PREFIX = "bias"

    def __init__(self, module: MLPClassifier | MLPRegressor):
        self.module = module
        self._dims_initialized = False  # TODO: Check if the scikit object has this.

    ####################################################################################

    # noinspection PyMethodMayBeStatic
    @t.final
    def kind(self) -> FrameworkKind:
        return "scikit"

    def get_params(self) -> Params:
        params = []
        for i in range(self._n_layers):
            params.append((f"{self.WEIGHT_KEY_PREFIX}_{i}", self.module.coefs_[i]))
            params.append((f"{self.BIAS_KEY_PREFIX}_{i}", self.module.intercepts_[i]))
        return OrderedDict(params)

    def set_params(self, params: Params):
        param_keys = list(params.keys())
        layer_nums = set(map(lambda txt: int(txt.split("_")[-1]), param_keys))
        num_layers = max(layer_nums) + 1

        weights = []
        biases = []
        for i in range(num_layers):
            w_i = params[f"{self.WEIGHT_KEY_PREFIX}_{i}"]
            b_i = params[f"{self.BIAS_KEY_PREFIX}_{i}"]
            weights.append(w_i)
            biases.append(b_i)

        self.module.coefs_ = weights
        self.module.intercepts_ = biases

    ####################################################################################

    @property
    def _n_layers(self) -> int:
        n = len(self.module.coefs_)
        if n != len(self.module.intercepts_):
            raise ValueError(
                "ScikitModule :: Inconsistent number of layers between "
                "coefficients/weights and intercepts/biases."
            )

        return n


class ScikitTrainer(AbstractTrainer):
    def __init__(
        self,
        node: Node | None = None,
        *,
        max_epochs: int | None = 5,
        partial: bool = False,
    ):
        super().__init__(node)
        self._partial = partial
        self._max_epochs = max_epochs
        self._first_partial_fit = True

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, module: ScikitModule, data: ScikitDataModule) -> list[Record]:
        """
        Fits (or trains) a Scikit-Learn module on a given data module.

        Args:
            module (ScikitModule): The module to fit.
            data (ScikitDataModule): The data module to use for training.

        Returns:
            A list of records containing the results of the training.
        """
        inputs, targets = data.train_data(self.node)
        if len(inputs) != len(targets):
            raise ValueError(
                f"Number of 'inputs'({len(inputs)}) does not match number of "
                f"'targets'({len(targets)})."
            )

        if self._partial:
            classes = None
            if self._first_partial_fit:
                classes = np.unique(targets)
                self._first_partial_fit = False

            for _ in range(self._max_epochs):
                module.module.partial_fit(inputs, targets, classes)
        else:
            if self._max_epochs is not None:
                module.module.max_iter = self._max_epochs

            module.module.fit(inputs, targets)

        return self._extract_records(module, mode="train")

    def test(self, module: ScikitModule, data: ScikitDataModule) -> list[Record]:
        """
        Tests a module on a given data module by loading its testing data.

        Args:
            module (ScikitModule): The module to test.
            data (ScikitDataModule): The data module to use for testing.

        Returns:
            A list of records containing the results of the testing.
        """
        return []

    def validate(self, module: ScikitModule, data: ScikitDataModule) -> list[Record]:
        """
        Evaluates (or tests) a module on a given data module by loading its
        validation data.

        Args:
            module (ScikitModule): The module to validate.
            data (ScikitDataModule): The data module to use for validation.

        Returns:
            A list of records containing the results of the validation.
        """
        return []

    @staticmethod
    def _extract_records(
        module: ScikitModule,
        mode: t.Literal["train", "test", "valid"],
    ) -> list[Record]:
        records = []
        for step, loss in enumerate(module.module.loss_curve_):
            records.append({f"{mode}/step": step, f"{mode}/loss": loss})
            # TODO: Change keys (^^)
        return records
