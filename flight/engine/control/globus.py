import typing as t
from concurrent.futures import Future

import globus_compute_sdk

from ...federation.topologies import Node
from .base import AbstractController


class GlobusController(AbstractController):
    _globus_compute_executor: globus_compute_sdk.Executor | None = None

    def __init__(self):
        super().__init__()
        if self._globus_compute_executor is None:
            self._globus_compute_executor = globus_compute_sdk.Executor()

    def __call__(self, fn: t.Callable, /, **kwargs) -> Future:
        if not isinstance(self._globus_compute_executor, globus_compute_sdk.Executor):
            raise ValueError("Executor is not a Globus Computer Executor.")

        if "node" not in kwargs:
            raise KeyError(
                f"{self.__name__.__class__} requires keyword `node` to be provided."
            )

        node = kwargs["node"]
        if not isinstance(node, Node):
            raise ValueError(f"Illegal value {type(node)=} != `Node`.")

        self._globus_compute_executor.endpoint_id = node.globus_comp_id
        future = self._globus_compute_executor.submit(fn, **kwargs)
        return future

    def shutdown(self) -> None:
        self._globus_compute_executor.shutdown()
