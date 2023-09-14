from concurrent.futures import Future

from flox.flock import FlockNode
from flox.learn.backends import FloxExecutor


class GlobusComputeExecutor(FloxExecutor):
    """
    Class that executes tasks on Globus Compute.
    """

    def __init__(self):
        super().__init__()

    def submit(self, fn, node: FlockNode, /, *args, **kwargs) -> Future:
        import globus_compute_sdk as globus_compute

        endpoint_id = node.globus_compute_endpoint
        print(f"{endpoint_id=}")
        with globus_compute.Executor(endpoint_id) as gce:
            future = gce.submit(fn, node, *args, **kwargs)
        return future

    def collect(self):
        pass
