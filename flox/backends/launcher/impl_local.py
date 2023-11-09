from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future

from flox.flock import FlockNode
from flox.backends.launcher.impl_base import Launcher


class LocalLauncher(Launcher):
    """
    Class that launches tasks locally.
    """

    def __init__(self, pool: str, n_workers: int = 1):
        super().__init__()
        self.n_workers = n_workers
        if pool == "process":
            self.pool = ProcessPoolExecutor(n_workers)
        elif pool == "thread":
            self.pool = ThreadPoolExecutor(n_workers)
        else:
            raise ValueError(
                "Illegal value for argument `pool`. Must be either 'pool' or 'thread'."
            )

    def submit(self, fn, node: FlockNode, /, *args, **kwargs) -> Future:
        return self.pool.submit(fn, node, *args, **kwargs)

    def collect(self):
        pass
