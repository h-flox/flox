from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import torch

from flox.learn.typing import Params, Loss
from flox.strategies.commons.averaging import average_state_dicts
from flox.topos import NodeState, NodeID, AggrState, WorkerState

if t.TYPE_CHECKING:
    from flox.runtime import JobResult
    from flox.strategies.aggregator import AggregatorStrategy
    from flox.strategies.client import ClientStrategy
    from flox.strategies.trainer import TrainerStrategy
    from flox.strategies.worker import WorkerStrategy


class DefaultClientStrategy:
    def select_worker_nodes(
        self, state: NodeState, workers: t.Iterable[NodeID], seed: int | None = None
    ) -> t.Iterable[NodeID]:
        return workers


class DefaultAggregatorStrategy:
    def round_start(self):
        pass

    def aggregate_params(
        self,
        state: AggrState,
        children_states: t.Mapping[NodeID, NodeState],
        children_state_dicts: t.Mapping[NodeID, Params],
        **kwargs,
    ) -> Params:
        return average_state_dicts(children_state_dicts, weights=None)

    def round_end(self):
        pass


class DefaultWorkerStrategy:
    def work_start(self, state: WorkerState) -> WorkerState:
        return state

    def before_training(
        self, state: WorkerState, data: t.Any
    ) -> tuple[WorkerState, t.Any]:
        return state, data

    def after_training(
        self, state: WorkerState, optimizer: torch.optim.Optimizer
    ) -> WorkerState:
        return state

    def work_end(self, result: JobResult) -> JobResult:
        return result


class DefaultTrainerStrategy:
    def trainer_kwargs(self) -> dict[str, t.Any]:
        return {}

    def before_backprop(self, state: WorkerState, loss: Loss) -> Loss:
        return loss

    def after_backprop(self, state: WorkerState, loss: Loss) -> Loss:
        return loss


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
    """Implementation of callbacks specific to the execution of the training loop on the worker nodes."""

    def __repr__(self) -> str:
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
        yield from strategies


class DefaultStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__(
            DefaultClientStrategy(),
            DefaultAggregatorStrategy(),
            DefaultWorkerStrategy(),
            DefaultTrainerStrategy(),
        )
