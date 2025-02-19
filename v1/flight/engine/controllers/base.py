from __future__ import annotations

import abc
import typing as t

if t.TYPE_CHECKING:
    from concurrent.futures import Future

    from v1.flight.types import P, T


class AbstractController(abc.ABC):
    @abc.abstractmethod
    def submit(self, fn: t.Callable[P, T], /, **kwargs) -> Future[T]:  # noqa
        """
        Executes a given function with the provided keyword arguments.

        Args:
            fn (typing.Callable[P, T]): A function to be executed by the controller.
            **kwargs: Keyword arguments to be passed to the function.

        Returns:
            Future object representing the asynchronous execution of the function.

        Notes:
            Controllers can *only* accept keyword arguments to pass into the functions.
        """

    @abc.abstractmethod
    def shutdown(self) -> None:
        """
        Properly and safely shuts down a controller.
        """
