import typing as t

import torch

from flight.strategies.aggr import AggrStrategy
from flight.strategies.base import DefaultStrategy, Strategy
from flight.strategies.coord import CoordStrategy
from flight.strategies.trainer import TrainerStrategy
from flight.strategies.worker import WorkerStrategy

Loss: t.TypeAlias = torch.Tensor
Params: t.TypeAlias = dict[str, torch.Tensor]
NodeState: t.TypeAlias = t.Any


def load_strategy(strategy_name: str, **kwargs) -> Strategy:
    assert isinstance(strategy_name, str), "`strategy_name` must be a string."
    match strategy_name.lower():
        case "default":
            return DefaultStrategy()

        case "fedasync" | "fed-async":
            from flight.strategies.impl.fedasync import FedAsync

            return FedAsync(**kwargs)

        case "fedavg" | "fed-avg":
            from flight.strategies.impl.fedavg import FedAvg

            return FedAvg(**kwargs)

        case "fedprox" | "fed-prox":
            from flight.strategies.impl.fedprox import FedProx

            return FedProx(**kwargs)

        case "fedsgd" | "fed-sgd":
            from flight.strategies.impl.fedsgd import FedSGD

            return FedSGD(**kwargs)
        case _:
            raise ValueError(f"Strategy '{strategy_name}' is not recognized.")


__all__ = [
    "AggrStrategy",
    "Strategy",
    "CoordStrategy",
    "TrainerStrategy",
    "WorkerStrategy",
]
