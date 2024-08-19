from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor

from flox.federation.jobs import Job
from flox.runtime.launcher.base import Launcher


class LocalLauncher(Launcher):
    """
    Launcher implementation that processes jobs locally.
    """

    pool: Executor

    def __init__(self, pool: str, n_workers: int = 1):
        super().__init__()
        self.max_workers = n_workers
        match pool:
            case "federation":
                self.pool = ProcessPoolExecutor(n_workers)
            case "thread":
                self.pool = ThreadPoolExecutor(n_workers)
            case _:
                raise ValueError(
                    "Illegal value for argument `pool`. Must be either 'pool' or 'thread'."
                )

    def submit(self, job: Job, /, **kwargs) -> Future:
        # TODO: Adjust this typing (i.e., Future is not always returned in the case where `max_workers == 1`.
        #       Then clarify the logic behind this.
        return self.pool.submit(job, **kwargs)  # type: ignore
        # if self.max_workers > 1:
        #     return self.pool.submit(fn, node, *args, **kwargs)
        # else:
        #     return fn(node, *args, **kwargs)

    def collect(self):
        pass
