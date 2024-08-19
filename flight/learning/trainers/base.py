import typing as t

from flight.learning.modules.prototypes import DataLoadable
from flight.learning.modules import HasParameters


class Trainer(t.Protocol):
    """
    Object class that is responsible for training `Trainable` objects.
    """

    def fit(self, model: HasParameters, data: DataLoadable, *args, **kwargs):
        """
        fit

        Args:
            model (HasParameters):
            data (DataLoadable):
            *args:
            **kwargs:

        Returns:

        """

    def test(self, model: HasParameters, *args, **kwargs):
        """
        test

        Args:
            model:
            *args:
            **kwargs:

        Returns:

        """

    def validate(self, model: HasParameters, data: DataLoadable, *args, **kwargs):
        """
        evaluate

        Args:
            model:
            *args:
            **kwargs:

        Returns:

        """
