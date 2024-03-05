import typing as t

from flox.strategies.aggregator import AggregatorStrategy
from flox.strategies.client import ClientStrategy
from flox.strategies.trainer import TrainerStrategy
from flox.strategies.worker import WorkerStrategy


class Strategy(t.NamedTuple):
    """
    A strategy...
    """

    client_strategy: ClientStrategy | None = None
    """..."""
    aggr_strategy: AggregatorStrategy | None = None
    """..."""
    worker_strategy: WorkerStrategy | None = None
    """..."""
    trainer_strategy: TrainerStrategy | None = None
    """..."""
