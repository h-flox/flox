from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future, Executor

from flox.flock import FlockNode
from flox.jobs import NodeCallable
from flox.runtime.launcher.base import Launcher


class LocalLauncher(Launcher):
    """
    Launcher implementation that processes jobs locally.
    """

    pool: Executor

    def __init__(self, pool: str, n_workers: int = 1):
        super().__init__()
        self.n_workers = n_workers
        match pool:
            case "process":
                self.pool = ProcessPoolExecutor(n_workers)
            case "thread":
                self.pool = ThreadPoolExecutor(n_workers)
            case _:
                raise ValueError(
                    "Illegal value for argument `pool`. Must be either 'pool' or 'thread'."
                )

    def submit(self, fn: NodeCallable, node: FlockNode, /, *args, **kwargs) -> Future:
        # TODO: Adjust this typing (i.e., Future is not always returned in the case where `n_workers == 1`.
        #       Then clarify the logic behind this.
        return self.pool.submit(fn, node, *args, **kwargs)
        # if self.n_workers > 1:
        #     return self.pool.submit(fn, node, *args, **kwargs)
        # else:
        #     return fn(node, *args, **kwargs)

    def collect(self):
        pass
