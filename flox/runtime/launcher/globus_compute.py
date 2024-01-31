from concurrent.futures import Future
from typing import Any, Callable

from flox.flock import FlockNode
from flox.runtime.launcher.base import Launcher


class GlobusComputeLauncher(Launcher):
    """
    Class that executes tasks on Globus Compute.
    """

    def __init__(self):
        super().__init__()

    def submit(
        self, fn: Callable[[FlockNode, ...], Any], node: FlockNode, /, *args, **kwargs
    ) -> Future:
        import globus_compute_sdk as globus_compute

        endpoint_id = node.globus_compute_endpoint
        with globus_compute.Executor(endpoint_id=endpoint_id) as gce:
            future = gce.submit(fn, node, *args, **kwargs)
        return future

    def collect(self):
        pass