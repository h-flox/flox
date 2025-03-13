import typing as t

from .events import CoordinatorEvents, AggregatorEvents, WorkerEvents


class CoordinatorState:
    round: int  # 1-based, the first iteration is 1
    seed: int | None
    times: dict[str, t.Any]


class AggregatorState:
    """
    ...
    """

    times: dict[str, float | None]
    """
    ...
    """

    def __init__(self) -> None:
        self.times = {
            AggregatorEvents.STARTED.name: None,
            AggregatorEvents.COMPLETED.name: None,
        }


class WorkerState:
    ignite: None

    times: dict[str, float | None]

    def __init__(self) -> None:
        self.times = {
            WorkerEvents.STARTED.name: None,
            WorkerEvents.COMPLETED.name: None,
        }


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
                        # keys: Events.EPOCH_COMPLETED.name and Events.COMPLETED.name iteration
"""
