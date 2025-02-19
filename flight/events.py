from __future__ import annotations

import enum
import typing as t

if t.TYPE_CHECKING:
    ...


# TODO: Add some quantifies (e.g., "every") similar to the events in Ignite.
class CoordEvent(enum.Flag):
    ROUND_START = enum.auto()
    BEFORE_AGGR = enum.auto()
    BEFORE_TEST_DATA_LOAD = enum.auto()
    AFTER_TEST_DATA_LOAD = enum.auto()
    AFTER_AGGR = enum.auto()
    ROUND_COMPLETED = enum.auto()


class AggrEvent(enum.Flag):
    AGGREGATION_START = enum.auto()
    AGGREGATION_COMPLETED = enum.auto()


class WorkerEvent(enum.Flag):
    WORKER_START = enum.auto()
    WORKER_COMPLETED = enum.auto()
