from __future__ import annotations

import abc
import typing as t


class _EnforceSuperMeta(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        if not getattr(instance, "_initialized", False):
            raise RuntimeError(
                f"{cls.__name__}.__init__() must call super().__init__()"
            )
        return instance

class AggregationPolicy(t.Protocol):
    def __call__(*args, **kwargs):
        pass


class WorkerSelectionPolicy(t.Protocol):
    def __call__(*args, **kwargs):
        pass


class Strategy:
    def __init__(self):
        pass
