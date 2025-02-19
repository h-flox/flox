from __future__ import annotations

import typing as t

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings  # noqa

from v1.flight.learning import AbstractTrainer
from v1.flight.learning.scikit.data import ScikitDataModule
from v1.flight.learning.scikit.module import ScikitModule
from v1.flight.topologies import Node
from v1.flight.types import Record


class ScikitTrainer(AbstractTrainer):
    def __init__(
        self,
        node: Node | None = None,
        *,
        max_epochs: int = 5,  # TODO: Convert to const.
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
