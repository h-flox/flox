import typing as t
from concurrent.futures import Future

if t.TYPE_CHECKING:
    pass


class GlobusCP:
    def __call__(self, fn: t.Callable, /, *args, **kwargs) -> Future:
        pass
