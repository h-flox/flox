import typing as t
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

if t.TYPE_CHECKING:
    from concurrent.futures import Executor


class LocalCP:
    executor: Executor

    def __init__(self, mode: str | t.Literal["process", "thread"], **kwargs):
        match mode:
            case "process":
                self.executor = ProcessPoolExecutor(**kwargs)
            case "thread":
                self.executor = ThreadPoolExecutor(**kwargs)

    def __call__(self, fn: t.Callable, /, *args, **kwargs) -> Future:
        return self.executor.submit(fn, *args, **kwargs)

    def shutdown(self):
        self.executor.shutdown(wait=True, cancel_futures=False)
