"""
```mermaid
sequenceDiagram
    participant Coordinator
    participant Aggregator
    participant Worker
    
    Coordinator->>Aggregator
    Coordinator->>Worker
``` 
"""

from __future__ import annotations

import functools
import inspect
import typing as t
from collections.abc import Iterable

from ignite.engine.events import EventEnum, Events, EventsList


class FlightEventEnum(EventEnum):
    """
    Parent class of events specific to launching federations in Flight.

    This is the parent class of the following child classes:

    - [`CoordinatorEvents`][flight.events.CoordinatorEvents]
    - [`AggregatorEvents`][flight.events.AggregatorEvents]
    - [`WorkerEvents`][flight.events.WorkerEvents]

    This class simply extends the
    [`EventEnum`](https://pytorch.org/ignite/
    generated/ignite.engine.events.EventEnum.html#ignite.engine.events.EventEnum)
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

    def __or__(self, other: t.Any) -> EventsList:
        return EventsList() | self | other

    def __hash__(self) -> int:
        return hash(f"{self.__class__.__name__}.{self._name_}")


class CoordinatorEvents(FlightEventEnum):
    """
    Coordinator events.

    ```mermaid
    flowchart LR
        a-->b
    ```
    """

    STARTED = "started"
    """
    Triggered at the start of a federation.
    """
    COMPLETED = "completed"
    """
    Triggered at the end of a federation.
    """

    ROUND_STARTED = "round_started"
    """
    Triggered at the start of a federation round.
    """
    ROUND_COMPLETED = "round_completed"
    """
    Triggered at the end of a federation round.
    """

    WORKER_SELECTION_STARTED = "worker_selection_started"
    """
    Triggered at the start of worker selection.
    """
    WORKER_SELECTION_COMPLETED = "worker_selection_ended"
    """
    Triggered Occurs at the end of worker selection.
    """

    GET_DATA_STARTED = "get_data_started"
    """
    Triggered Occurs at the start of data loading
    (typically test data on the coordinator).
    """
    GET_DATA_COMPLETED = "get_data_completed"
    """
    Triggered Occurs at the end of data loading
    (typically test data on the coordinator).
    """

    EXCEPTION_RAISED = "exception_raised"
    """
    Triggered when an exception is raised.
    """


class AggregatorEvents(FlightEventEnum):
    """
    Aggregator events.
    """

    STARTED = "started"
    """
    Triggered at the start of an aggregation job.
    """
    COMPLETED = "completed"
    """
    Triggered at the end of an aggregation job.
    """


class WorkerEvents(FlightEventEnum):
    """
    Worker events.
    """

    STARTED = "started"
    """
    Triggered at the start of a worker's job.
    """
    COMPLETED = "completed"
    """
    Triggered at the end of a worker's.
    """


IgniteEvents = Events
"""
Simple, and more clear, alias to 
[`Events`](https://pytorch.org/ignite/generated/ignite.engine.events.Events.html)
in PyTorch Ignite that are used during model training.
"""

GenericEvents: t.TypeAlias = t.Union[
    CoordinatorEvents,
    AggregatorEvents,
    WorkerEvents,
    IgniteEvents,
]
"""
A union type of all event types usable in Flight (defined in Flight and Ignite).
"""

EventHandler: t.TypeAlias = t.Callable[
    [dict[str, t.Any]],  # context
    None,
]
"""
Callable definition for functions that are called on the firing of an event.
"""


_ON_DECORATOR_META_FLAG: t.Final[str] = "_event_type"
"""
Name of the flag used in the [`on()`][flight.events.on] decorator.
"""


# TODO: Add a `priority` functionality on each event so that we can
#       sort the event ordering by priority. This would look like:
#       `@on(WorkerEvents.STARTED(priority=2))`.
def on(event_type: GenericEvents | EventsList):
    """
    Decorator function that wraps a function with the given `event_type`.

    This is used to class methods in [`Strategy`][flight.strategy.Strategy]
    classes where methods can use this function to decorate class methods
    to fire for specific event types.

    ```python
    class SomeObject:
        ...

        @on(CoordinatorEvents.STARTED)
        def greet(self, context) -> None:
            print("Start")
    ```
    """

    def decorator(func: EventHandler):
        setattr(func, _ON_DECORATOR_META_FLAG, event_type)  # Store metadata

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def fire_event_handler_by_type(
    obj: t.Any,
    event_type: GenericEvents | EventsList,
    context: dict[str, t.Any] | None = None,
    logger: t.Any = None,
) -> None:
    """
    Fires the event handler implementations for a single event type or a list of
    event types.

    Args:
        obj (t.Any): Object that has attributes with event handlers which are
            decorated by the [`on()`][flight.events.on] decorator.
        event_type (GenericEvents | EventsList): The event type(s) to fire with
            the given context.
        context (dict[str, typing.Any] | None): Optional context that the event
            handler is run with. Defaults to `None`.
        logger: Optional logger to use for logging the event firing.

    Notes:
        The order in which event handlers for `event_type` is not guaranteed.
        Ensure that the logic of your given `Strategy` for federated learning
        with Flight does not rely on a certain order of these event handlers
        to run.
    """
    if context is None:
        context = {}

    for _name, handler in get_event_handlers(obj, event_type):
        handler(context)


def get_event_handlers(
    obj: t.Any,
    event_type: GenericEvents | EventsList,
    predicate: t.Callable[..., bool] | None = None,
) -> list[tuple[str, EventHandler]]:
    """
    Given an object with implemented `EventHandler`s, this function returns a list
    of `EventHandlers` specified by the `event_type`.

    It is worth noting that the `event_type` argument can specify a specific type
    of event or a list of events via the `|` operator.

    Args:
        obj (t.Any): Object that contains event handlers within its definitions.
        event_type (GenericEvents | EventsList): Event type(s) to get the handlers
            for.
        predicate (t.Callable[[...], bool] | None): A filtering method used by
            `inspect.getmembers` to filter out members of `obj` to search for
            `EventHandler`s. Defaults to `None`; which will set
            `predicate=inspect.ismethod`.

    Returns:
        List of tuples with the `EventHandler`s in `obj`. Each tuple in the list
            contains the name of the `EventHandler` and the callable function
            itself.

    Examples:
        >>> obj = ...
        >>> handlers = get_event_handlers(obj, WorkerEvents.STARTED)
        >>> handlers = get_event_handlers(
        >>>     obj,
        >>>     WorkerEvents.STARTED | WorkerEvents.COMPLETED,
        >>> )
    """
    if predicate is None:
        predicate = inspect.ismethod

    def get_type_matching(met: t.Any) -> tuple[str, ...]:
        matching = []

        if isinstance(event_type, EventsList):
            matching.append("list")
        else:  # isinstance(event_type, GenericEvents) == True
            matching.append("generic")

        if isinstance(met, EventsList):
            matching.append("list")
        else:  # isinstance(met, GenericEvents) == True
            matching.append("generic")

        return tuple(matching)

    handlers = []
    for name, method in inspect.getmembers(obj, predicate=predicate):
        method_event_type = getattr(method, _ON_DECORATOR_META_FLAG, None)
        if method_event_type is None:
            continue

        match get_type_matching(method_event_type):
            case "generic", "generic":
                if event_type == method_event_type:
                    handlers.append((name, method))

            case "generic", "list":
                if event_type in method_event_type:
                    handlers.append((name, method))

            case "list", "generic":
                if method_event_type in event_type:  # type: ignore[operator]
                    handlers.append((name, method))

            case "list", "list":
                et_set = set(event_type)  # type: ignore[arg-type]
                met_set = set(method_event_type)  # type: ignore[arg-type]
                if et_set.intersection(met_set):
                    handlers.append((name, method))

            case _:
                raise TypeError("Illegal types for `event_type`.")

    return handlers


def get_event_handlers_by_genre(
    obj: t.Any,
    event_genre: type[EventEnum] | t.Iterable[type[EventEnum]],
    predicate: t.Callable[..., bool] | None = None,
) -> list[tuple[str, EventHandler]]:
    """
    Given an object, get all of its event handler attributes that are designated
    to be run for a given genre of events (e.g.,
    [`WorkerEvents`][flight.events.WorkerEvents],
    [`AggregatorEvents`][flight.events.AggregatorEvents],
    [`CoordinatorEvents`][flight.events.CoordinatorEvents],
    [`Events`](https://pytorch.org/ignite/generated/ignite.engine.events.Events.html)
    /[`IgniteEvents`][flight.events.IgniteEvents]).

    Args:
        obj (t.Any): ...
        event_genre (type[EventEnum] | typing.Iterable[type[EventEnum]]):
            ...
        predicate (typing.Callable[..., bool | None): ...

    Returns:
        List of tuples with the `EventHandler`s in `obj` belonging to the given
            genre(s). Each tuple in the list contains the name of the
            `EventHandler` and the callable function itself.

    Examples:
        >>> from flight.events import on, WorkerEvents
        >>>
        >>> class MyObject:
        >>>    @on(WorkerEvents.STARTED)
        >>>    def foo(self, context):
        >>>        print("Hello, world!")
        >>>
        >>> get_event_handlers_by_genre(obj, WorkerEvents)
        ["foo", <bound method MyStrategy.foo of ...]
    """
    handlers: list[tuple[str, EventHandler]] = []
    if predicate is None:
        predicate = inspect.ismethod

    #######################################################################

    def _process_single_event_enum(_event_type: type[EventEnum]) -> None:
        """
        Check for and add (if any) event handlers for the given event type.
        """
        for name, method in inspect.getmembers(obj, predicate):
            method_event_type = getattr(method, _ON_DECORATOR_META_FLAG, None)

            print(f"{method_event_type=}  |  {_event_type=}")

            if isinstance(method_event_type, EventsList):
                if _event_type in method_event_type:  # type: ignore
                    handlers.append((name, method))
            elif isinstance(method_event_type, GenericEvents):  # type: ignore
                if isinstance(method_event_type, _event_type):
                    handlers.append((name, method))
            # if isinstance(method_event_type, _event_type):
            #     handlers.append((name, method))

    def _process_iterable_of_event_enums(
        _event_types: t.Iterable[type[EventEnum]],
    ) -> None:
        """
        Check for and add (if any) event handlers for the given
        iterable set of event types.
        """
        for _event_type in _event_types:
            _process_single_event_enum(_event_type)

    #######################################################################

    if inspect.isclass(event_genre):
        if issubclass(event_genre, EventEnum):
            _process_single_event_enum(event_genre)
        else:
            raise TypeError

    elif isinstance(event_genre, Iterable):
        is_subclass_mask: list[bool] = [issubclass(e, EventEnum) for e in event_genre]
        if all(is_subclass_mask):
            _process_iterable_of_event_enums(event_genre)
        else:
            raise ValueError

    return handlers
