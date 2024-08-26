import typing as t
from concurrent.futures import Future

from .base import AbstractController


class SerialController(AbstractController):
    """
    A simple controller that runs jobs serially.

    This class should be used for rapid prototyping and simple debugging on
    your local machine.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, fn: t.Callable, /, **kwargs) -> Future:
        future: Future = Future()
        try:
            result = fn(**kwargs)
            future.set_result(result)
        except Exception as exception:
            future.set_exception(exception)
        return future

    def shutdown(self):
        return None
