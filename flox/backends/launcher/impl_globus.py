from concurrent.futures import Future

import globus_compute_sdk

from flox.backends.launcher.impl_base import Launcher, LauncherFunction
from flox.flock import FlockNode


class GlobusComputeLauncher(Launcher):
    """
    Class that executes tasks on Globus Compute.
    """

    def __init__(self):
        super().__init__()
        if self._globus_compute_executor is None:
            self._globus_compute_executor = globus_compute_sdk.Executor()

    def submit(
        self, fn: LauncherFunction, node: FlockNode, /, *args, **kwargs
    ) -> Future:
        endpoint_id = node.globus_compute_endpoint
        self._globus_compute_executor.endpoint_id = endpoint_id
        future = self._globus_compute_executor.submit(fn, node, *args, **kwargs)
        return future

    def collect(self):
        pass
