from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future

from flox.learn.backends.base import FloxExecutor


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
