import typing as t

from flight.learning.datasets import DataLoadable
from flight.learning.modules import Trainable


class Trainer(t.Protocol):
    """
    Object class that is responsible for training `Trainable` objects.
    """

    def fit(self, model: Trainable, data: DataLoadable, *args, **kwargs):
        """
        fit

        Args:
            model (Trainable):
            data (DataLoadable):
            *args:
            **kwargs:

        Returns:

        """

    def test(self, model: Trainable, *args, **kwargs):
        """
        test

        Args:
            model:
            *args:
            **kwargs:

        Returns:

        """

    def validate(self, model: Trainable, data: DataLoadable, *args, **kwargs):
        """
        evaluate

        Args:
            model:
            *args:
            **kwargs:

        Returns:

        """
