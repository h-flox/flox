from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from concurrent.futures import Future

T = t.TypeVar("T")


class ControlPlane(t.Protocol):
    """
    A protocol that defines an interface for a control plane object.

    The definition of this protocol is based on the `submit` method found in
    the [`concurrent.futures.Executor`](https://docs.python.org/3/
    library/concurrent.futures.html#concurrent.futures.Executor.submit)
    function definition.
    """

    def submit(self, fn: t.Callable[..., T], /, *args, **kwargs) -> Future[T]:
        """
        Executes a given function with the provided keyword arguments.

        Args:
            fn (typing.Callable): A function to be executed by the controller.
            *args: Arguments to be passed to the function.
            **kwargs: Keyword arguments to be passed to the function.

        Returns:
            Future object representing the asynchronous execution of the function.
        """
