import typing as t


class ControlPlane(t.Protocol):
    def submit(self, fn: t.Callable, /, *args, **kwargs):
        pass
