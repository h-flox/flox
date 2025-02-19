from __future__ import annotations

import abc
import typing as t

from numpy._typing import ArrayLike

from v1.flight.learning import AbstractDataModule
from v1.flight.topologies import Node


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
