from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from flox.strategies.aggregator import AggregatorStrategy
from flox.strategies.client import ClientStrategy
from flox.strategies.trainer import TrainerStrategy
from flox.strategies.worker import WorkerStrategy


class DefaultClientStrategy(ClientStrategy):
    pass
    # def __init__(self):
    #     super().__init__(self)


class DefaultAggregatorStrategy(AggregatorStrategy):
    pass
    # def __init__(self):
    #     super().__init__(self)


class DefaultWorkerStrategy(WorkerStrategy):
    pass
    # def __init__(self):
    #     super().__init__(self)


class DefaultTrainerStrategy(TrainerStrategy):
    pass
    # def __init__(self):
    #     super().__init__(self)


@dataclass(frozen=True, repr=False)
class Strategy:
    """
    A ``Strategy`` implementation is made up of a set of implementations for strategies on each part of the
    topology during execution.
    """

    client_strategy: ClientStrategy = field(default_factory=DefaultClientStrategy)
    """Implementation of callbacks specific to the CLIENT node."""
    aggr_strategy: AggregatorStrategy = field(default_factory=DefaultAggregatorStrategy)
    """Implementation of callbacks specific to the AGGREGATOR nodes."""
    worker_strategy: WorkerStrategy = field(default_factory=DefaultWorkerStrategy)
    """Implementation of callbacks specific to the WORKER nodes."""
    trainer_strategy: TrainerStrategy = field(default_factory=DefaultTrainerStrategy)
    """Implementation of callbacks specific to the training loop on the worker nodes."""

    # def __post_init__(self):
    #     if self.client_strategy is not None:
    #         self.client_strategy

    def __repr__(self):
        return str(self)

    def __str__(self) -> str:
        name = self.__class__.__name__
        inner = ", ".join(
            [
                f"{strategy_key}={strategy_value.__class__.__name__}"
                for (strategy_key, strategy_value) in iter(self)
                if strategy_value is not None
            ]
        )
        return f"{name}({inner})"

    def __iter__(self) -> t.Iterator[tuple[str, t.Any]]:
        strategies = (
            ("client_strategy", self.client_strategy),
            ("aggr_strategy", self.aggr_strategy),
            ("worker_strategy", self.worker_strategy),
            ("trainer_strategy", self.trainer_strategy),
        )
        for strategy_key, strategy_value in strategies:
            yield strategy_key, strategy_value
