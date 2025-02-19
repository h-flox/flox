"""

```py
>>> class MyStrategy(Strategy):
>>>     def __init__(self, aggreagtion_policy, selection_policy) -> None:
>>>         super().__init__(aggregation_policy, selection_policy)
>>>
>>>     @on(EventType.START)
>>>     def my_start_task(self):
>>>         print("My task started!")
>>>
>>> strat = MyStrategy(...)
>>> strat.get_event_handlers(EvenType.START)
[(my_start_task, <function MyStrategy.my_start_task at ...)]
```

"""

import abc
import inspect
import typing as t
from enum import auto, Enum

from v1.flight import Topology

from v1.flight.federation_v2.events import Event

Context: t.TypeAlias = dict[str, t.Any]
FlightEventHandler: t.TypeAlias = t.Callable[[Context], None]


class EventType(Enum):
    FOO = auto()
    BAR = auto()


class AggregationPolicy(abc.ABC):
    def aggregate(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


class WorkerSelectionPolicy(abc.ABC):
    def select(self, topology: Topology, *args, **kwargs):
        pass

    def __call__(self, topology: Topology, *args, **kwargs):
        pass


class _EnforceSuperMeta(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)  # Calls __init__
        if not getattr(instance, "_initialized", False):
            raise RuntimeError(
                f"{cls.__name__}.__init__() must call super().__init__()"
            )
        return instance


class Strategy(metaclass=_EnforceSuperMeta):
    aggregation_policy: AggregationPolicy
    selection_policy: WorkerSelectionPolicy

    def __init__(
        self,
        aggregation_policy: AggregationPolicy | None = None,
        selection_policy: WorkerSelectionPolicy | None = None,
    ) -> None:
        super().__init__()
        self._initialized = True  # Mark that super().__init__() was called
        self._required_attrs = ["aggregation_policy", "selection_policy"]

        if aggregation_policy:
            self.aggregation_policy = aggregation_policy
        if selection_policy:
            self.selection_policy = selection_policy

        for attr in self._required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"`{self.__class__.__name__}` missing `{attr}` implementation."
                )

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        if not hasattr(cls, "_initialized"):
            raise RuntimeError(
                f"`{cls.__name__}.__init__()` must call `super().__init__()`."
            )
        return instance

    def aggregate(self, *args, **kwargs):
        """
        Shorthand method to use the aggregation method defined by the
        aggregation policy.
        """
        return self.aggregation_policy(*args, **kwargs)

    def select(self, topology: Topology, *args, **kwargs):
        """
        Shorthand method to use the worker selection method defined by the
        selection policy.
        """
        return self.selection_policy(topology, *args, **kwargs)

    @t.final
    def get_event_handlers(
        self, event_type: Event
    ) -> list[tuple[str, FlightEventHandler]]:
        handlers = []
        for name, method in inspect.getmembers(self, predicate=inspect.isfunction):
            method_event = getattr(method, "_event_type", None)
            if method_event and (method_event & event_type):
                handlers.append((name, method))
        return handlers

    @t.final
    def get_event_handlers_by_type(
        self, event_type: type[Event]
    ) -> list[tuple[str, FlightEventHandler]]:
        handlers = []
        for name, method in inspect.getmembers(self, predicate=inspect.isfunction):
            method_event = getattr(method, "_event_type", None)
            if method_event and isinstance(method_event, event_type):
                handlers.append((name, method))
        return handlers

    @t.final
    def fire_event(
        self,
        event_type: Event,
        context: Context,
    ) -> None:
        for name, method in self.get_event_handlers(event_type):
            method(context)

    @t.final
    def fire_events_by_type(
        self,
        event_type: Event,
        context: Context,
        # logger: Logger,
    ):
        for name, method in self.get_event_handlers(event_type):
            method(context)
