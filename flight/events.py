# from __future__ import annotations

# import enum
# import typing as t

# if t.TYPE_CHECKING:
#     ...


# # TODO: Add some quantifies (e.g., "every") similar to the events in Ignite.
# class CoordEvent(enum.Flag):
#     ROUND_START = enum.auto()
#     BEFORE_AGGR = enum.auto()
#     BEFORE_TEST_DATA_LOAD = enum.auto()
#     AFTER_TEST_DATA_LOAD = enum.auto()
#     AFTER_AGGR = enum.auto()
#     ROUND_COMPLETED = enum.auto()


# class AggrEvent(enum.Flag):
#     AGGREGATION_START = enum.auto()
#     AGGREGATION_COMPLETED = enum.auto()


# class WorkerEvent(enum.Flag):
#     WORKER_START = enum.auto()
#     WORKER_COMPLETED = enum.auto()

"""
Hello?
"""

import typing as t

from ignite.engine.events import EventEnum, EventsList


class FlightEventEnum(EventEnum):
    """
    Parent class of events specific to launching federations in Flight.

    This is the parent class of the following child classes:

    - [`CoordinatorEvents`][flight.events.CoordinatorEvents]
    - [`AggregatorEvents`][flight.events.AggregatorEvents]
    - [`WorkerEvents`][flight.events.WorkerEvents]

    This class simply extends the [`EventEnum`](https://pytorch.org/ignite/generated/ignite.engine.events.EventEnum.html#ignite.engine.events.EventEnum)
    class provided by PyTorch-Ignite. The only difference between
    `FlightEventEnum` and Ignite's `EventEnum` is that instances of a
    `FlightEventEnum` can only be equal if they are of the same class.
    For instance:

    ```python
    >>> from flight.events import *
    >>> e1 = CoordinatorEvents.STARTED
    >>> e2 = WorkerEvents.STARTED
    >>> e1 == e2
    False
    ```

    In the above example, if `CoordinatorEvents`, `AggregatorEvents`,
    and `WorkerEvents` had only extended Ignite's `EventEnum`, then
    `e1 == e2` would evaluate to `True`, which is not useful for Flight.
    We want to discern between events belonging to these classes, even if
    they share the same name—which is done for simplicity.
    """

    def __eq__(self, other: t.Any) -> bool:
        if self.__class__ != other.__class__:
            return False
        return EventEnum.__eq__(self, other)


class CoordinatorEvents(FlightEventEnum):
    """
    Coordinator events.
    """

    STARTED = "started"
    """Triggered at the start of a federation."""
    COMPLETED = "completed"
    """Triggered at the end of a federation."""

    ROUND_STARTED = "round_started"
    """Triggered at the start of a federation round."""
    ROUND_COMPLETED = "round_completed"
    """Triggered at the end of a federation round."""

    WORKER_SELECTION_STARTED = "worker_selection_started"
    """Triggered at the start of worker selection."""
    WORKER_SELECTION_COMPLETED = "worker_selection_ended"
    """Triggered Occurs at the end of worker selection."""

    GET_DATA_STARTED = "get_data_started"
    """Triggered Occurs at the start of data loading (typically test data on the coordinator)."""
    GET_DATA_COMPLETED = "get_data_completed"
    """Triggered Occurs at the end of data loading (typically test data on the coordinator)."""

    EXCEPTION_RAISED = "exception_raised"
    """Triggered when an exception is raised."""

    def __or__(self, other: t.Any) -> EventsList:
        return EventsList() | self | other


class AggregatorEvents(FlightEventEnum):
    """
    Aggregator events.
    """

    STARTED = "started"
    """Triggered at the start of an aggregation job."""
    COMPLETED = "completed"
    """Triggered at the end of an aggregation job."""


class WorkerEvents(FlightEventEnum):
    """
    Worker events.
    """

    STARTED = "started"
    """Triggered at the start of a worker's job."""
    COMPLETED = "completed"
    """Triggered at the end of a worker's."""


class CoordinatorState:
    ...


class AggregatorState:
    ...


class WorkerState:
    ...
