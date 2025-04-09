from __future__ import annotations

import functools
import typing as t
from enum import Flag, auto

from ignite import engine

if t.TYPE_CHECKING:
    pass


class CoordEvent(Flag):
    ROUND_START = auto()
    BEFORE_AGGR = auto()
    BEFORE_TEST_DATA_LOAD = auto()
    AFTER_TEST_DATA_LOAD = auto()
    AFTER_AGGR = auto()
    ROUND_END = auto()


class AggrEvent(Flag):
    BEFORE_AGGR = auto()
    AFTER_AGGR = auto()


class WorkerEvent(Flag):
    BEFORE_TRAIN = auto()
    AFTER_TRAIN = auto()


IgniteEvent: t.TypeAlias = engine.Events
IgniteEventHandler: t.TypeAlias = t.Callable  # [[Engine, t.Any], None]

Event: t.TypeAlias = IgniteEvent | CoordEvent | AggrEvent | WorkerEvent
EventHandler: t.TypeAlias = t.Callable  # [[Context], None]


# TODO: We need to double-check that this does not conflict with Ignite's own
#       event system.
def on(event_type: Event):
    def decorator(func):
        setattr(func, "_event_type", event_type)  # Store metadata

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator
