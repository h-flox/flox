from flight.strategies.aggr import AggrStrategy
from flight.strategies.base import Strategy, DefaultStrategy
from flight.strategies.coord import CoordStrategy
from flight.strategies.trainer import TrainerStrategy
from flight.strategies.worker import WorkerStrategy


def load_strategy(strategy_name: str, **kwargs) -> Strategy:
    assert NotImplementedError


__all__ = [
    "AggrStrategy",
    "Strategy",
    "CoordStrategy",
    "TrainerStrategy",
    "WorkerStrategy",
]
