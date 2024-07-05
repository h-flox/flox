from concurrent.futures import Future


class SerialCP:
    def __call__(self, fn, /, *args, **kwargs) -> Future:  # noqa
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future
