import itertools
import pytest

from flight.events import *


def test_non_overlapping_event_names():
    starts = [
        CoordinatorEvents.STARTED,
        AggregatorEvents.STARTED,
        WorkerEvents.STARTED,
    ]
    for s1, s2 in itertools.permutations(starts, 2):
        assert s1 == s1
        assert s1 != s2
        assert s1.value == s2.value

    completes = [
        CoordinatorEvents.COMPLETED,
        AggregatorEvents.COMPLETED,
        WorkerEvents.COMPLETED,
    ]
    for c1, c2 in itertools.permutations(completes, 2):
        assert c1 == c1
        assert c1 != c2
        assert c1.value == c2.value


def test_event_binary_operations():
    items = CoordinatorEvents.STARTED | AggregatorEvents.STARTED

    assert isinstance(items, EventsList)

    assert CoordinatorEvents.STARTED in items
    assert AggregatorEvents.STARTED in items

    assert WorkerEvents.STARTED not in items


def test_get_event_handlers():
    class MyClass:
        """Strategy-like class."""

        def __init__(self):
            super().__init__()

        @on(WorkerEvents.STARTED)
        def cache(self, context):
            context["tmp"] = True

        @on(CoordinatorEvents.STARTED | CoordinatorEvents.COMPLETED)
        def greet(self, context):
            print("Hello!")

        @on(WorkerEvents.COMPLETED)
        def cleanup(self, context):
            del context["tmp"]

    # Ensure that the `get_event_handler()` method correctly returns event
    # handlers for the specified event type.
    start_handlers = get_event_handlers(MyClass(), WorkerEvents.STARTED)
    assert len(start_handlers) == 1
    assert start_handlers[0][0] == "cache"

    completed_handlers = get_event_handlers(MyClass(), WorkerEvents.COMPLETED)
    assert len(completed_handlers) == 1
    assert completed_handlers[0][0] == "cleanup"

    coord_start_handlers = get_event_handlers(MyClass(), CoordinatorEvents.STARTED)
    coord_compl_handlers = get_event_handlers(MyClass(), CoordinatorEvents.COMPLETED)
    coord_start_compl_handlers = get_event_handlers(
        MyClass(),
        CoordinatorEvents.STARTED | CoordinatorEvents.COMPLETED,
    )

    assert len(coord_start_handlers) == 1
    assert len(coord_compl_handlers) == 1
    assert len(coord_start_compl_handlers) == 1

    # Ensure that `get_event_handler()` method handles the case of `EventList`
    # objects which are given by the `|` (or the `__or__`) operator.
    start_and_comp_handlers = get_event_handlers(
        MyClass(),
        WorkerEvents.STARTED | WorkerEvents.COMPLETED,
    )
    assert len(start_and_comp_handlers) == 2
    assert {name for name, _ in start_and_comp_handlers} == {"cache", "cleanup"}

    # Ensure that event handlers that do not exist in `MyClass` are not returned.
    other_handlers = get_event_handlers(MyClass(), AggregatorEvents.STARTED)
    assert len(other_handlers) == 0

    # Ensure that context/state is properly passed through the events.
    context = {}
    start_handlers[0][1](context)
    assert context["tmp"] == True

    completed_handlers[0][1](context)
    assert "tmp" not in context


def test_get_event_handlers_by_genre():
    class MyClass:
        """Strategy-like class."""

        def __init__(self):
            super().__init__()

        @on(WorkerEvents.STARTED)
        def cache(self, context):
            context["tmp"] = True

        @on(WorkerEvents.COMPLETED)
        def cleanup(self, context):
            del context["tmp"] 

        @on(CoordinatorEvents.STARTED)
        def coord_hello(self, context):
            print("Coordinator: 'Hello!'")

        @on(AggregatorEvents.STARTED | AggregatorEvents.COMPLETED)
        def greet(self, context):
            print("Hello!")

    instance = MyClass()
    
    worker_handlers = get_event_handlers_by_genre(instance, WorkerEvents)
    aggr_handlers = get_event_handlers_by_genre(instance, AggregatorEvents)
    coord_handlers = get_event_handlers_by_genre(instance, CoordinatorEvents)

    worker_handler_names = set(h[0] for h in worker_handlers)
    aggr_handler_names = set(h[0] for h in aggr_handlers)
    print(aggr_handlers)
    coord_handler_names = set(h[0] for h in coord_handlers)
    
    assert worker_handler_names == {"cache", "cleanup"}
    assert aggr_handler_names == {"greet"}
    assert coord_handler_names == {"coord_hello"}
    
    worker_coord_handlers = get_event_handlers_by_genre(instance, [WorkerEvents, CoordinatorEvents])
    worker_coord_handler_names = set([h[0] for h in worker_coord_handlers])

    assert worker_coord_handler_names == {"cache", "cleanup", "coord_hello"}
