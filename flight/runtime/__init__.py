from __future__ import annotations

__all__ = ["Runtime"]

import typing as t

if t.TYPE_CHECKING:
    from concurrent.futures import Future

    from .control.base import ControlPlane
    from .data.base import DataPlane


def make_default_control_plane() -> ControlPlane:
    from concurrent.futures import ThreadPoolExecutor

    return ThreadPoolExecutor(max_workers=1)


def make_default_data_plane() -> DataPlane:
    return InMemoryDataPlane()


class Runtime:
    control_plane: ControlPlane
    """
    The control plane is responsible for submitting functions to be executed at the
    appropriate compute resource (e.g., remote endpoint, node in an HPC system,
    local thread).
    """

    data_plane: DataPlane
    """
    The data plane is responsible for facilitating data transfer needed for
    federated learning with Flight.
    """

    def __init__(
        self,
        control_plane: ControlPlane | None = None,
        data_plane: DataPlane | None = None,
    ):
        if control_plane is None:
            control_plane = make_default_control_plane()

        if data_plane is None:
            data_plane = make_default_data_plane()

        self.control_plane = control_plane
        self.data_plane = data_plane

    def submit(self, fn: t.Callable, /, *args, **kwargs) -> Future:
        """
        Shorthand method for submitting a function to be executed by the control plane.

        Args:
            fn (typing.Callable): The function to be executed.
            *args: Arguments to be passed to the function
            **kwargs: Keyword arguments to be passed to the function.

        Returns:
            A future object with the (pending) result of the function execution.
        """
        return self.control_plane.submit(fn, *args, **kwargs)

    def transfer(self, data: t.Any) -> t.Any:
        """
        Shorthand method for data transfer using the data plane.

        Args:
            data (typing.Any): The data to be transferred.

        Returns:
            A reference to the data after transfer.
        """
        return self.data_plane.transfer(data)

    @classmethod
    def simple_setup(
        cls,
        max_workers: int = 1,
        exec_kind: t.Literal["thread", "process"] = "process",
    ) -> Runtime:
        """
        Simple setup for a `Runtime` object with a control plane and data plane.

        - The control plane is a `ThreadPoolExecutor` or `ProcessPoolExecutor` with
          the specified number of workers.
        - The data plane is an in-memory data plane.

        Args:
            max_workers (int): The maximum number of workers for the control plane.
                Defaults to 1.
            exec_kind (str): The kind of executor to use. Either "thread" or "process".
                Defaults to "process".

        Returns:

        """
        from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor

        control_plane: Executor
        match exec_kind:
            case "thread":
                control_plane = ThreadPoolExecutor(max_workers=max_workers)
            case "process":
                control_plane = ProcessPoolExecutor(max_workers=max_workers)
            case _:
                raise ValueError(f"Unsupported exec_kind: {exec_kind}")

        return cls(control_plane=control_plane, data_plane=make_default_data_plane())


class InMemoryDataPlane:
    """
    The default data plane implementation that leaves data in memory.

    More clearly, its `transfer` method does nothing to the data but return it as is.
    """

    # noinspection PyMethodMayBeStatic
    def transfer(self, data: t.Any) -> t.Any:
        """
        Returns the data as is.

        Args:
            data (typing.Any): Data to be transferred.

        Returns:
            Inputted data as given.
        """
        return data
