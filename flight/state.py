from __future__ import annotations

import abc
import typing as t
from dataclasses import dataclass, field

from .events import AggregatorEvents, CoordinatorEvents, WorkerEvents

if t.TYPE_CHECKING:
    from datetime import datetime


########################################################################################


@dataclass
class AbstractNodeState(abc.ABC):
    def __new__(cls, *args, **kwargs):
        if cls == AbstractNodeState:  # or cls.__bases__[0] == AbstractNodeState:
            raise TypeError("Cannot instantiate abstract class.")
        return super().__new__(cls)


########################################################################################


def _make_coord_times() -> dict[str, datetime | None]:
    return {event.name: None for event in CoordinatorEvents}  # noqa


@dataclass
class CoordinatorState(AbstractNodeState):
    round: int  # 1-based, the first iteration is 1
    times: dict[t.Any, datetime | None] = field(default_factory=_make_coord_times)
    seed: int | None = field(default=None)


########################################################################################


def _make_aggr_times() -> dict[str, datetime | None]:
    return {event.name: None for event in AggregatorEvents}  # noqa


@dataclass
class AggregatorState(AbstractNodeState):
    """
    ...
    """

    times: dict[str, datetime | None] = field(default_factory=_make_aggr_times)
    """
    ...
    """


########################################################################################


def _make_worker_times() -> dict[str, datetime | None]:
    return {event.name: None for event in WorkerEvents}  # noqa


@dataclass
class WorkerState(AbstractNodeState):
    ignite: None = field(default=None)  # TODO
    """
    The state from training with Ignite the model on this worker. For more information
    of what is available in the Ignite state, see:
    [https://pytorch.org/ignite/generated/ignite.engine.events.State.html]()
    """

    times: dict[str, datetime | None] = field(default_factory=_make_worker_times)
    """
    The times at which events occurred during training on this worker. The keys are
    the names of the events that occurred, and the values are the times at which the
    events occurred.
    """


"""
https://pytorch.org/ignite/generated/ignite.engine.events.State.html
`ignite.engine.events.State` attributes:

state.iteration         # 1-based, the first iteration is 1
state.epoch             # 1-based, the first epoch is 1
state.seed              # seed to set at each epoch
state.dataloader        # data passed to engine
state.epoch_length      # optional length of an epoch
state.max_epochs        # number of epochs to run
state.batch             # batch passed to `process_function`
state.output            # output of `process_function` after a single iteration
state.metrics           # dictionary with defined metrics if any
state.times             # dictionary with total and per-epoch times fetched on
                        # keys: Events.EPOCH_COMPLETED.name and
                        # Events.COMPLETED.name iteration
"""
