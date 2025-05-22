from __future__ import annotations

import abc
import inspect
import typing as t

from .events import get_event_handlers, get_event_handlers_by_genre
from .learning.module import Params

if t.TYPE_CHECKING:
    from ignite.engine.events import EventsList

    from .events import EventEnum, EventHandler, GenericEvents
    from .system import Topology
    from .system.types import NodeID


_SUPER_META_FLAG: t.Final[str] = "__super_is_initialized"
"""
Flag to ensure that the call to `super().__init__()` is called via
[`_EnforceSuperMeta`][flight.strategy._EnforceSuperMeta].
"""


class _EnforceSuperMeta(abc.ABCMeta):
    """
    Metaclass that requires instances of classes that use it
    (via `class MyClass(metaclass=_EnforceSuperMeta): ...`) to use
    `super().__init__()` in the initializer.
    """

    def __call__(cls, *args, **kwargs):
        """
        ...
        """
        instance = super().__call__(*args, **kwargs)
        if not getattr(instance, _SUPER_META_FLAG, False):
            raise AttributeError(
                f"{cls.__name__}.__init__() must call super().__init__()"
            )
        return instance


class AggregationPolicy(t.Protocol):
    """
    ...
    """

    # TODO
    def __call__(self, modules: dict[NodeID, Params], *args, **kwargs):
        """
        ...
        """


class WorkerSelectionPolicy(t.Protocol):
    """
    Definition of a worker selection policy function / callable.
    """

    # TODO
    def __call__(self, topo: Topology) -> list[NodeID]:
        """
        ...
        """


class Strategy(metaclass=_EnforceSuperMeta):
    """
    A `Strategy` is a collection of logical pieces that are run during the
    execution of a federation.

    There are **two** logical pieces that must be included in a `Strategy`
    implementation:

    1. **Aggregation Policy**: A callable [object][flight.strategy.AggregationPolicy]
       that defines how models are aggregated in a `Strategy` during federation.

    2. **Worker Selection Policy**: A callable
       [object][flight.strategy.WorkerSelectionPolicy] that defines how
       a `Strategy` should do worker selection in a given federation round.

    These logical pieces can be included directly by passing them to
    `Strategy.__init__()` or by defining them explicitly within the `Strategy`
    definition. Below is an example of both approaches:

    ```python
    from flight.strategy import Strategy

    # OPTION 1: Directly passing in required logic through the initializer.
    def aggr_policy(*args, **kwargs): ...
    def selection_policy(*args, **kwargs): ...
    strategy = Strategy(aggr_policy, selection_policy)

    # OPTION 2: Providing required logic through the `Strategy` definition.
    class MyStrategy(Strategy):
        def aggregation_policy(self, *args, **kwargs): ...
        def selection_policy(self, *args, **kwargs): ...
    ```

    Additional, optional pieces of logic can be included as _event handlers_. These
    are included within the project definition (similar to option 2 in the above
    code sample).

    An example of how to include these optional pieces of logic is below:

    ```python
    from flight.events import *

    class MyStrategy(Strategy):
        def __init__(self):
            super().__init__(...) # pass required logic

        @on(WorkerEvents.STARTED):
        def say_hello(context):
            r = context.state.round  # Access inherited state from context.
            print(f"Hello, this is the start of round #{r}.")
    ```
    """

    def __init__(
        self,
        aggregation_policy: AggregationPolicy | None = None,
        selection_policy: WorkerSelectionPolicy | None = None,
    ) -> None:
        """
        Args:
            aggregation_policy (AggregationPolicy | None): Callable
                object/function that defines how model parameters are
                aggregated by aggregator nodes and the coordinator in
                the `Topology` used in a federation. Defaults to `None`.
            selection_policy (WorkerSelectionPolicy | None): Callable
                object/function that defines how worker nodes are selected
                to perform local training at each aggregation round.
                Defaults to `None`.
        """
        super().__init__()
        setattr(self, _SUPER_META_FLAG, True)

        if aggregation_policy:
            self.aggregation_policy = aggregation_policy
        if selection_policy:
            self.selection_policy = selection_policy

        for attr in self._required_attrs():
            if not hasattr(self, attr):
                raise AttributeError(
                    f"`{self.__class__.__name__}` missing `{attr}` implementation."
                )

    def aggregate(self, *args, **kwargs):
        """
        Shorthand method to use the aggregation method defined by the
        aggregation policy.
        """
        return self.aggregation_policy(*args, **kwargs)

    def select_workers(self, topology: Topology, *args, **kwargs) -> list[NodeID]:
        """
        Shorthand method to use the worker selection method defined by the
        worker selection policy.
        """
        return self.selection_policy(topology, *args, **kwargs)

    @classmethod
    def _required_attrs(cls) -> tuple[str, str]:
        """
        Defines/returns attributes that must be defined by the user.

        Returns:
            Tuple of attribute names that are required to be implemented/provided
            by the `Strategy`.
        """
        return "aggregation_policy", "selection_policy"

    #################################################################################

    def fire_event_handler(
        self,
        event_type: GenericEvents | EventsList,
        context: dict[str, t.Any] | None = None,
    ) -> None:
        """
        Fires the event handler implementations for a single event type or a list of
        event types.

        Args:
            event_type (GenericEvents | EventsList): The event type(s) to fire with
                the given context.
            context (dict[str, typing.Any] | None): Optional context that the event
                handler is run with. Defaults to `None`.

        Notes:
            The order in which event handlers for `event_type` is not guaranteed.
            Ensure that the logic of your given `Strategy` for federated learning
            with Flight does not rely on a certain order of these event handlers
            to run.
        """

        # NOTE: Should this be in the Federation instead? Need to think this over.
        #       Remember, this has to run on the different nodes
        #       (coordinator/aggregator/workers)
        if context is None:
            print("Creating a new context")
            context = {}
        else:
            print("Inherited a context")

        for _name, handler in get_event_handlers(self, event_type):
            handler(context)

    @t.final
    def get_event_handlers(
        self,
        event_type: GenericEvents | EventsList,
    ) -> list[tuple[str, EventHandler]]:
        """
        Returns all the implemented event handlers included in a `Strategy` that are
        marked to be for `event_type` by its decorator.

        Args:
            event_type (GenericEvents | EventList): The type of event to (or list of
                event types) to grab the event handlers for.

        Returns:
            List of event handlers meant for the provided `event_type`.

        Notes:
            This code is shorthand for
            [`get_event_handlers()`][flight.events.get_event_handlers]
            that simply passes `self` as the initial argument (`obj`).
        """
        return get_event_handlers(
            self,
            event_type,
            predicate=inspect.ismethod,
        )

    @t.final
    def get_event_handlers_by_genre(
        self,
        event_genre: type[EventEnum] | t.Iterable[type[EventEnum]],
    ):
        """
        Returns all the implemented event handlers in a `Strategy` that are
        to be run for a given genre of events (e.g.,
        [`WorkerEvents`][flight.events.WorkerEvents],
        [`AggregatorEvents`][flight.events.AggregatorEvents],
        [`CoordinatorEvents`][flight.events.CoordinatorEvents],
        [`Events`](
            https://pytorch.org/ignite/generated/ignite.engine.events.Events.html
        )
        /[`IgniteEvents`][flight.events.IgniteEvents]).

        Args:
            event_genre (type[EventEnum] | typing.Iterable[type[EventEnum]]):
                The genre of events to grab the event handlers for.

        Returns:
            List of tuples with the `EventHandler`s in `obj` belonging to the given
                genre(s). Each tuple in the list contains the name of the
                `EventHandler` and the callable function itself.

        Notes:
            This method is shorthand for the [`get_event_handlers_by_genre()`]
            [flight.events.get_event_handlers_by_genre] function that simply passes
            `self` as the initial argument (`obj`).

        Examples:
            >>> from flight.events import on, WorkerEvents
            >>> from flight.strategy import DefaultStrategy
            >>>
            >>> class MyStrategy(DefaultStrategy):
            >>>    @on(WorkerEvents.STARTED)
            >>>    def foo(self, context):
            >>>        print("Hello, world!")
            >>>
            >>> s = MyStrategy(...)
            >>> s.get_event_handlers_by_genre(WorkerEvents)
            ["foo", <bound method MyStrategy.foo of ...]
        """
        return get_event_handlers_by_genre(
            self,
            event_genre,
            predicate=inspect.ismethod,
        )


class DefaultStrategy(Strategy):
    """
    Simple strategy implementation used for convenience.
    """

    def __init__(self):
        super().__init__()

    def aggregation_policy(self, *args, **kwargs):
        return  # TODO: Change this when we have this working in `fitter.py`.

    def selection_policy(self, topo: Topology, *args, **kwargs):
        # TODO: Change this when we have this working in `fitter.py`.
        return topo.workers
