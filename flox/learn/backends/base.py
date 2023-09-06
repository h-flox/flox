from abc import ABC
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor


class FloxExecutor(ABC):
    """
    Base class for executing functions in an FL process.
    """

    def __init__(self):
        pass

    def submit(self, fn, /, *args, **kwargs) -> Future:
        raise NotImplementedError()

    def collect(self):
        pass


class GlobusComputeExecutor(FloxExecutor):
    """
    Class that executes tasks on Globus Compute.
    """

    def __init__(self):
        super().__init__()

    def submit(self, fn, /, *args, **kwargs):
        import globus_compute_sdk as globus_compute

        endpoint_id = None
        with globus_compute.Executor(endpoint_id) as gce:
            future = gce.submit(fn, *args, **kwargs)
        return future

    def collect(self):
        pass


class LocalExecutor(FloxExecutor):
    """
    Class that executes tasks locally.
    """

    def __init__(self, pool: str, n_workers: int = 1):
        super().__init__()
        self.n_workers = n_workers
        if pool == "pool":
            self.pool = ProcessPoolExecutor(n_workers)
        elif pool == "thread":
            self.pool = ThreadPoolExecutor(n_workers)
        else:
            raise ValueError(
                "Illegal value for argument `pool`. "
                "Must be either 'pool' or 'thread'."
            )

    def submit(self, fn, /, *args, **kwargs) -> Future:
        return self.pool.submit(fn, *args, **kwargs)

    def collect(self):
        pass
