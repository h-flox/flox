from concurrent.futures import Future

from flox.learn.backends.base import FloxExecutor


class GlobusComputeExecutor(FloxExecutor):
    """
    Class that executes tasks on Globus Compute.
    """

    def __init__(self):
        super().__init__()

    def submit(self, fn, /, *args, **kwargs) -> Future:
        import globus_compute_sdk as globus_compute

        endpoint_id = None
        with globus_compute.Executor(endpoint_id) as gce:
            future = gce.submit(fn, *args, **kwargs)
        return future

    def collect(self):
        pass
