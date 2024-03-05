from __future__ import annotations

import typing as t

import globus_compute_sdk

from flox.runtime.launcher.base import Launcher

if t.TYPE_CHECKING:
    from concurrent.futures import Future
    from flox.flock import FlockNode
    from flox.jobs import NodeCallable


class GlobusComputeLauncher(Launcher):
    """
    Class that executes tasks on Globus Compute.
    """

    _globus_compute_executor: globus_compute_sdk.Executor | None = None

    def __init__(self):
        super().__init__()
        # if self._globus_compute_executor is None:
        #     self._globus_compute_executor = globus_compute_sdk.Executor()

    def submit(
        self,
        fn: NodeCallable,
        node: FlockNode,
        /,
        *args,
        **kwargs,
    ) -> Future:
        assert isinstance(self._globus_compute_executor, globus_compute_sdk.Executor)
        assert node.globus_compute_endpoint is not None
        self._globus_compute_executor.endpoint_id = node.globus_compute_endpoint
        future = self._globus_compute_executor.submit(fn, node, *args, **kwargs)
        return future

    def collect(self):
        pass
