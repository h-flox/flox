from __future__ import annotations

import typing as t
from concurrent.futures import Future

import globus_compute_sdk

from flight.engine.controllers.base import AbstractController
from flight.federation.topologies import Node

if t.TYPE_CHECKING:
    from flight.types import P, T


class GlobusController(AbstractController):
    """
    Implementation of a controller that uses the Globus Compute SDK to execute functions
    on remote endpoints.
    """

    _globus_compute_executor: globus_compute_sdk.Executor | None = None

    def __init__(self):
        super().__init__()
        if self._globus_compute_executor is None:
            self._globus_compute_executor = globus_compute_sdk.Executor()

    def submit(self, fn: t.Callable[P, T], /, **kwargs) -> Future[T]:  # noqa
        """
        Submits a function to be executed on a remote endpoint using the
        [Globus Compute](https://globus-compute.readthedocs.io/en/latest/index.html)
        Function-as-a-Service (FaaS) platform.

        Args:
            fn (Callable[P, T]): Function to be executed on the remote endpoint.
            **kwargs: Keyword arguments to be passed to the function.

        Returns:
            Future object representing the asynchronous execution of the function.

        Notes:
            - Globus Compute enforces a data transfer limit of roughly 5MB per function.
              To avoid this limitation, it is recommended to use
              [ProxyStore](https://docs.proxystore.dev/).
            - Arguments to function can *only* be passed via keyword arguments.
        """
        if not isinstance(self._globus_compute_executor, globus_compute_sdk.Executor):
            raise ValueError("Executor is not a Globus Computer Executor.")

        if "node" not in kwargs:
            raise KeyError(
                f"{self.__class__.__name__} requires keyword `node` to be provided."
            )

        node = kwargs["node"]
        if not isinstance(node, Node):
            raise ValueError(f"Illegal value {type(node)=} != `Node`.")

        self._globus_compute_executor.endpoint_id = node.globus_comp_id
        future = self._globus_compute_executor.submit(fn, **kwargs)
        return future

    def shutdown(self) -> None:
        """
        Shuts down the Globus Compute Executor.
        """
        if self._globus_compute_executor is not None:
            self._globus_compute_executor.shutdown()
        else:
            # TODO: Log warning.
            pass
