from __future__ import annotations

import pydantic as pyd
import typing as t
import functools

from flight.strategies.commons.averaging import average_state_dicts

if t.TYPE_CHECKING:
    import torch
    from numpy.random import Generator

    NodeState: t.TypeAlias = t.Any
    NodeID: t.TypeAlias = int | str
    Params: t.TypeAlias = t.Any
    Loss: t.TypeAlias = torch.Tensor

    from flight.federation.topologies.node import Node
    from flight.federation.jobs.result import Result


class DefaultCoordStrategy:
    def select_workers(
        self, state: NodeState, children: t.Iterable[Node], rng: Generator
    ) -> t.Sequence[Node]:
        return children


class DefaultAggrStrategy:
    def start_round(self):
        pass

    def aggregate_params(
        self,
        state: NodeState,
        children: t.Mapping[NodeID, NodeState],
        children_state_dicts: t.Mapping[NodeID, Params],
        **kwargs,
    ) -> Params:
        return average_state_dicts(children_state_dicts, weights=None)

    def end_round(self):
        pass


class DefaultWorkerStrategy:
    def start_work(self, state: NodeState) -> NodeState:
        return state

    def before_training(
        self, state: NodeState, data: Params
    ) -> tuple[NodeState, Params]:
        return state, data

    def after_training(
        self, state: NodeState, optimizer: torch.optim.Optimizer
    ) -> NodeState:
        return state

    def end_work(self, result: Result) -> Result:
        return result


class DefaultTrainerStrategy:
    def before_backprop(self, state: NodeState, loss: Loss) -> Loss:
        return loss

    def after_backprop(self, state: NodeState, loss: Loss) -> Loss:
        return loss


@pyd.dataclasses.dataclass(frozen=True, repr=False)
class Strategy:
    coord_strategy: str = pyd.Field()
    aggr_strategy: str = pyd.Field()
    worker_strategy: str = pyd.Field()
    trainer_strategy: str = pyd.Field()

    def __iter__(self) -> t.Iterator[tuple[str, t.Any]]:
        yield from (
            ("coord_strategy", self.coord_strategy),
            ("aggr_strategy", self.aggr_strategy),
            ("worker_strategy", self.worker_strategy),
            ("trainer_strategy", self.trainer_strategy),
        )

    def __repr__(self) -> str:
        return str(self)

    @functools.cached_property
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


class DefaultStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__(
            coord_strategy=DefaultCoordStrategy(),
            aggr_strategy=DefaultAggrStrategy(),
            worker_strategy=DefaultWorkerStrategy(),
            trainer_strategy=DefaultTrainerStrategy(),
        )
