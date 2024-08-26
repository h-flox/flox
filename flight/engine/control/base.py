from __future__ import annotations

import abc
import typing as t

if t.TYPE_CHECKING:
    from concurrent.futures import Future


class AbstractController(abc.ABC):
    @abc.abstractmethod
    def __call__(self, fn: t.Callable, /, **kwargs) -> Future:
        """
        ...

        Args:
            fn (typing.Callable):
            *args:
            **kwargs:

        Returns:

        """

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Properly and safely shuts down a controller."""
