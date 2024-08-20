from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flight.learning.modules import HasParameters
    from flight.learning.modules.prototypes import DataLoadable
    from flight.types import Record


class Trainer(t.Protocol):
    """
    Object class that is responsible for training *trainable* objects.
    """

    def fit(
        self, model: HasParameters, data: DataLoadable, *args, **kwargs
    ) -> list[Record]:
        """
        fit

        Args:
            model (HasParameters): Trainable model to evaluate.
            data (DataLoadable): Object with loadable data to evaluate with.

        Returns:

        """

    def test(
        self, model: HasParameters, data: DataLoadable, *args, **kwargs
    ) -> list[Record]:
        """
        test

        Args:
            model (HasParameters): Trainable model to evaluate.
            data (DataLoadable): Object with loadable data to evaluate with.

        Returns:

        """

    def validate(
        self, model: HasParameters, data: DataLoadable, *args, **kwargs
    ) -> list[Record]:
        """
        evaluate

        Args:
            model (HasParameters): Trainable model to evaluate.
            data (DataLoadable): Object with loadable data to evaluate with.

        Returns:

        """
