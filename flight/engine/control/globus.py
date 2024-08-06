import typing as t
from concurrent.futures import Future

if t.TYPE_CHECKING:
    pass


class GlobusCP:
    def __call__(self, fn: t.Callable, /, *args, **kwargs) -> Future:
        # TODO: Implement this using Globus Compute.
        #       Current implementation is from `./serial.py`.
        future: Future = Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as exception:
            future.set_exception(exception)
        return future
