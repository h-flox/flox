from __future__ import annotations

import abc
import typing as t

if t.TYPE_CHECKING:
    from concurrent.futures import Future

    from flight.types import P, T


class AbstractController(abc.ABC):
    @abc.abstractmethod
    def __call__(self, fn: t.Callable[P, T], /, **kwargs) -> Future[T]:  # noqa
        """
        ...

        Args:
            fn (typing.Callable[P, T]): A function to be executed by the controller.
            **kwargs: Keyword arguments to be passed to the function.

        Returns:
            ...

        Notes:
            Controllers can *only* accept keyword arguments to pass into the functions.
        """

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Properly and safely shuts down a controller."""
