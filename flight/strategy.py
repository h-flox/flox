from __future__ import annotations

import abc
import inspect
import typing as t

from .events import get_event_handlers
from .system import Topology

if t.TYPE_CHECKING:
    from ignite.engine.events import EventsList

    from .events import EventHandler, GenericEvents


_SUPER_META_FLAG: t.Final[str] = "__super_is_initialized"
"""TODO"""


class _EnforceSuperMeta(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        if not getattr(instance, _SUPER_META_FLAG, False):
            raise RuntimeError(
                f"{cls.__name__}.__init__() must call super().__init__()"
            )
        return instance


class AggregationPolicy(t.Protocol):
    """..."""

    def __call__(self, *args, **kwargs):
        """..."""


class WorkerSelectionPolicy(t.Protocol):
    """..."""

    def __call__(self, *args, **kwargs):
        """..."""


class Strategy:
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
            aggregation_policy (AggregationPolicy | None): ...
            selection_policy (WorkerSelectionPolicy | None): ...
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

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        if not hasattr(cls, _SUPER_META_FLAG):
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

    def select_workers(self, topology: Topology, *args, **kwargs):
        """
        Shorthand method to use the worker selection method defined by the
        worker selection policy.
        """
        return self.selection_policy(topology, *args, **kwargs)

    @classmethod
    def _required_attrs(cls) -> tuple[str, ...]:
        """
        Defines/returns attributes that must be defined by the user.
        """
        return "aggregation_policy", "selection_policy"

    #################################################################################

    @t.final
    def get_event_handlers(
        self,
        event_type: GenericEvents | EventsList,
    ) -> list[tuple[str, EventHandler]]:
        """
        Returns all the implemented event handlers included in a `Strategy` that are
        marked to be for `event_type` by its decorator.

        Args:
            event_type (GenericEvents | EventList):

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
