from __future__ import annotations

import enum
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

    BEFORE_TRAINING = "before_training"
    """
    Triggered before the training loop starts with `Ignite` via `train_engine.run(...)`.
    """

    AFTER_TRAINING = "after_training"
    """
    Triggered after the training loop ends with `Ignite` via `train_engine.run(...)`.
    """

    COMPLETED = "completed"
    """
    Triggered at the end of a worker's.
    """


class TrainProcessFnEvents(FlightEventEnum):
    """
    Process function events for the training loop.

    These events are used to signal the start and end of various stages of the
    process function in PyTorch Ignite.

    Notes:

    """

    BATCH_PREPARE_STARTED = "batch_prepare_started"
    """
    Triggered before preparing a batch for training.
    """
    BATCH_PREPARE_COMPLETED = "batch_prepare_completed"
    """
    Triggered after preparing a batch for training.
    """

    BACKWARD_STARTED = "backward_started"
    """
    Triggered before backpropagation starts.
    """
    BACKWARD_COMPLETED = "backward_completed"
    """
    Triggered after backpropagation ends.
    """

    OPTIM_STEP_STARTED = "optim_step_started"
    """
    Triggered before an optimization step starts.
    """
    OPTIM_STEP_COMPLETED = "optim_step_completed"
    """
    Triggered after an optimization step ends.
    """


IgniteEvents: t.TypeAlias = Events
"""
Simple, and more clear, alias to
[`Events`](https://pytorch.org/ignite/generated/ignite.engine.events.Events.html)
in PyTorch Ignite that are used during model training.

Below is an overview of the events that exist within `Events`/`IgniteEvents`:

- `STARTED` :
  triggered when engine’s run is started
- `EPOCH_STARTED` :
  triggered when the epoch is started
- `GET_BATCH_STARTED` :
  triggered before next batch is fetched
- `GET_BATCH_COMPLETED` :
  triggered after the batch is fetched
- `ITERATION_STARTED` :
  triggered when an iteration is started
- `ITERATION_COMPLETED` :
  triggered when the iteration is ended
- `DATALOADER_STOP_ITERATION` :
  engine’s specific event triggered when dataloader has no more data to provide
- `EXCEPTION_RAISED` :
  triggered when an exception is encountered
- `TERMINATE_SINGLE_EPOCH` :
  triggered when the run is about to end the current epoch, after receiving
  a terminate_epoch() or terminate() call.
- `EPOCH_COMPLETED` :
  triggered when the epoch is ended. This is triggered even when terminate_epoch()
  is called, unless the flag skip_epoch_completed is set to True.
- `TERMINATE` :
  triggered when the run is about to end completely, after receiving terminate() call.
- `COMPLETED` :
  triggered when engine’s run is completed or terminated with terminate(), unless
  the flag skip_completed is set to True.
"""

GenericEvents: t.TypeAlias = t.Union[
    CoordinatorEvents,
    AggregatorEvents,
    WorkerEvents,
    TrainProcessFnEvents,
    IgniteEvents,
]
"""
A union type of all event types usable in Flight (defined in Flight and Ignite).
"""

Context: t.TypeAlias = dict[str, t.Any]
"""
Contextual information that is passed to event handlers when they are executed.
"""
EventHandler: t.TypeAlias = t.Callable[[Context], None]
"""
Callable definition for functions that are called on the firing of an event.
"""


_ON_DECORATOR_META_FLAG: t.Final[str] = "_event_type"
"""
Name of the flag used in the [`on()`][flight.events.on] decorator.
"""

_ON_DECORATOR_WHEN_FLAG: t.Final[str] = "_when_in_ignite"


class IgniteEventKinds(str, enum.Enum):
    TRAIN: str = "train"
    VALIDATE: str = "validate"
    TEST: str = "test"


def add_event_handler_to_obj(
    obj,
    event_type: GenericEvents | EventsList,
    handler: EventHandler,
    when: str | IgniteEventKinds | None = None,
):
    """
    Adds an event handler to an object with the given `event_type`.

    Args:
        obj (t.Any):
            Object to add the event handler to.
        event_type (GenericEvents | EventsList):
            The event type(s) to fire the decorated function for.
            This can be a single event type or a list of event types combined with
            the `|` operator.
        handler (EventHandler):
            The event handler function to add to the object. This function should
            accept a single argument, which is the context dictionary.
        when (str | IgniteEventKinds | None):
            Specifies whether the event handler is for training, validation,
            or testing. Defaults to `None`, which then defaults to
            `IgniteEventKinds.TRAIN` in [`on()`][flight.events.on].
    """
    handler_name: t.Final[str] = handler.__name__
    if hasattr(obj, handler_name):
        raise ValueError(
            f"Object `{obj}` already has an attribute named `{handler_name}`."
        )

    decorator = on(event_type, when=when)
    setattr(obj, handler_name, decorator(handler))


# TODO: Add a `priority` functionality on each event so that we can
#       sort the event ordering by priority. This would look like:
#       `@on(WorkerEvents.STARTED(priority=2))`.
def on(
    event_type: GenericEvents | EventsList,
    when: str | IgniteEventKinds | None = IgniteEventKinds.TRAIN,
) -> t.Callable[[EventHandler], t.Callable]:
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

    Args:
        event_type (GenericEvents | EventsList):
            The event type(s) to fire the decorated function for.
            This can be a single event type or a list of event types combined with
            the `|` operator.
        when (str | IgniteEventKinds):
            Specifies whether the event handler is for training, validation,
            or testing. Defaults to `IgniteEventKinds.TRAIN`.

    Notes:
        The `when` argument is _only_ useful when used for `IgniteEvents`.
        Specifically, it is used to specify whether the event handler is for
        training, validation, or testing.

        It defaults to `IgniteEventKinds.TRAIN` for all decorators.
        However, for clarity, you should always specify this `when` argument
        for `IgniteEvents` to avoid confusion.

        **Finally, do *not* use multiple `on()` decorators on the same class method.
        hat is not supported at this time. If you have repeat functionality for
        different events, we recommend the following:**

        ```python
        class SharedFunctionality:
            ...

            def shared_functionality(self, context):
                print("Shared functionality executed!")
                if "shared" in context:
                    context["shared"] += 1
                else:
                    context["shared"] = 1

            @on(IgniteEvents.STARTED, when="train")
            def train_test_started_1(self, context):
                self.shared_functionality(context)

            @on(IgniteEvents.STARTED, when="test")
            def train_test_started_2(self, context):
                self.shared_functionality(context)
        ```

        In the above example, we delegate the shared functionality to a separate
        class method and simply call that from separate decorated methods
        for the respective event types they are meant to activate for.
    """

    if when is None:
        when = IgniteEventKinds.TRAIN

    def decorator(func: EventHandler):
        setattr(func, _ON_DECORATOR_META_FLAG, event_type)  # Store flag metadata.
        setattr(func, _ON_DECORATOR_WHEN_FLAG, when)  # Only relevant to IgniteEvents.

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
        The order in which event handlers for `event_type` is _not_ guaranteed.
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
    when: str | IgniteEventKinds | None = None,
) -> list[tuple[str, EventHandler]]:
    """
    Given an object with implemented `EventHandler`s, this function returns a list
    of `EventHandlers` specified by the `event_type`.

    It is worth noting that the `event_type` argument can specify a specific type
    of event or a list of events via the `|` operator.

    Args:
        obj (t.Any):
            Object that contains event handlers within its definitions.
        event_type (GenericEvents | EventsList):
            Event type(s) to get the handlers for.
        predicate (t.Callable[[...], bool] | None):
            A filtering method used by `inspect.getmembers` to filter out members of
            `obj` to search for `EventHandler`s. Defaults to `None`; which will set
            `predicate=inspect.ismethod`.
        when (str | IgniteEventKinds | None):
            Specifies whether the event handler is for training, validation,
            or testing. Defaults to `None`. If not specified, all event handlers
            are returned regardless of their `when` value.

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

    if when is not None:
        # Filter out handlers that do not match the `when` argument.
        handlers = [
            (event, handler)
            for event, handler in handlers
            if getattr(handler, _ON_DECORATOR_WHEN_FLAG, None) == when
        ]

    return handlers


def get_event_handlers_by_genre(
    obj: t.Any,
    event_genre: type[EventEnum] | t.Iterable[type[EventEnum]],
    predicate: t.Callable[..., bool] | None = None,
    when: str | IgniteEventKinds | None = None,
) -> list[tuple[EventEnum, EventHandler]]:
    """
    Given an object, get all of its event handler attributes that are designated
    to be run for a given genre of events (e.g.,
    [`WorkerEvents`][flight.events.WorkerEvents],
    [`AggregatorEvents`][flight.events.AggregatorEvents],
    [`CoordinatorEvents`][flight.events.CoordinatorEvents],
    [`Events`](https://pytorch.org/ignite/generated/ignite.engine.events.Events.html)
    /[`IgniteEvents`][flight.events.IgniteEvents]).

    Args:
        obj (t.Any): Object that contains event handlers within its definitions.
        event_genre (type[EventEnum] | typing.Iterable[type[EventEnum]]):
            The type event genre(s) to get the handlers for. This can be a single
            event genre type (e.g., `WorkerEvents`) or an iterable of event genre
            types (e.g., given by `WorkerEvents | AggregatorEvents`).
        predicate (t.Callable[..., bool | None]):
            Passed to the `predicate` argument in the [`inspect.getmembers()`](
                https://docs.python.org/3.11/library/inspect.html#inspect.getmembers
            ) function.
        when (str | IgniteEventKinds | None):
            Specifies whether the event handler is for training, validation,
            or testing. Defaults to `None`. If not specified, all event handlers
            are returned regardless of their `when` value.

    Returns:
        List of tuples with the `EventHandler`s in `obj` belonging to the given
            genre(s). Each tuple in the list contains the name of the
            `EventHandler` and the callable function itself.

    Examples:
        This function allows you to quickly fetch all event handlers belonging to a
        genre of events. Below, we just have to indicate `WorkerEvents`:
        >>> from flight.events import on, WorkerEvents
        >>>
        >>> class MyObject:
        >>>    @on(WorkerEvents.STARTED)
        >>>    def foo(self, context):
        >>>        print("Hello, world!")
        >>>
        >>> get_event_handlers_by_genre(obj, WorkerEvents)
        [(WorkerEvents.STARTED, <bound method MyStrategy.foo of ...)]

        Additionally, if you wish to decorate event handlers to be run specifically for
        training, validation, or testing, you can use the `when` argument:
        >>> class MyClass:
        >>>     '''Strategy-like class.'''
        >>>
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>
        >>>     @on(IgniteEvents.STARTED, when="train")
        >>>     def train_started(self, context):
        >>>         print("Training started!")
        >>>
        >>>     @on(IgniteEvents.COMPLETED, when="test")
        >>>     def test_completed(self, context):
        >>>         print("Training completed!")
        >>>
        >>>     @on(IgniteEvents.STARTED, when="validate")
        >>>     def validate_started(self, context):
        >>>         print("Validation started!")
        >>>
        >>> instance = MyClass()
        >>> get_event_handlers_by_genre(obj, IgniteEvents, when="train")
        [(IgniteEvents.STARTED, <bound method MyClass.train_started of ...>)]

        It is worth noting that the above `when` argument is only relevant
        for `IgniteEvents` and is ignored for other event genres (e.g.,
        `WorkerEvents`, `AggregatorEvents`, and `CoordinatorEvents`).
    """
    handlers: list[tuple[EventEnum, EventHandler]] = []
    if predicate is None:
        predicate = inspect.ismethod

    #######################################################################

    def _process_single_event_enum(_genre: type[EventEnum]) -> None:
        """
        Check for and add (if any) event handlers for the given event type.
        """
        for _name, event_handler in inspect.getmembers(obj, predicate):
            event = getattr(event_handler, _ON_DECORATOR_META_FLAG, None)

            # If the event decorator is not set (i.e., not an event handler),
            # skip this method.
            if event is None:
                continue

            # Check if the event decorators for the event handler is an `EventsList`
            # (i.e., `|` operator was used to combine multiple event types).
            elif isinstance(event, EventsList):
                for e in event:
                    if isinstance(e, _genre):
                        handlers.append((e, event_handler))

            # Check if the event decorator for the event handler is a single
            # `EventEnum` type (namely belonging to `GenericEvents`). For note,
            # we do not explicitly check if `event` is an instance of `GenericEvents`
            # to make `mypy` happy.
            elif isinstance(event, _genre):
                handlers.append((event, event_handler))

    def _process_iterable_of_event_enums(
        _event_types: t.Iterable[type[EventEnum]],
    ) -> None:
        """
        Check for and add (if any) event handlers for the given
        iterable set of event types.
        """
        for _event_type in _event_types:
            _process_single_event_enum(_event_type)

    ####################################################################################

    if inspect.isclass(event_genre):
        if issubclass(event_genre, EventEnum):
            _process_single_event_enum(event_genre)
        else:
            raise TypeError("`event_genre` must be a subclass of `EventEnum`.")

    elif isinstance(event_genre, Iterable):
        is_subclass_mask: list[bool] = [issubclass(e, EventEnum) for e in event_genre]
        if all(is_subclass_mask):
            _process_iterable_of_event_enums(event_genre)
        else:
            raise ValueError("`event_genre` must be a subclass of `EventEnum`.")

    else:
        raise TypeError(
            "`event_genre` must be a subclass of `EventEnum` or an iterable "
            "of subclasses of `EventEnum`."
        )

    if when is not None:
        # Filter out handlers that do not match the `when` argument.
        handlers = [
            (event, handler)
            for event, handler in handlers
            if getattr(handler, _ON_DECORATOR_WHEN_FLAG, None) == when
        ]

    return handlers
