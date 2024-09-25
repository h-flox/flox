"""
```mermaid
classDiagram

class ClientStrategy {
    <<interface>>
    select_worker_nodes(self, state, children, seed) Iterable[Node]
}
```
"""

from flox.strategies.aggregator import AggregatorStrategy
from flox.strategies.client import ClientStrategy
from flox.strategies.strategy import DefaultStrategy, Strategy
from flox.strategies.trainer import TrainerStrategy
from flox.strategies.worker import WorkerStrategy


def load_strategy(strategy_name: str, **kwargs) -> Strategy:
    """
    Loads the strategy identified by the ``strategy_name`` argument.

    Notes:
        The argument `strategy_name` is *not* case-sensitive. Any value for that will be
        lower-cased via `strategy_name.lower()`.

    Raises:
        ValueError: in the event that the provided `strategy_name` is not supported.

    Args:
        strategy_name (str): The name of the strategy to be loaded.
        **kwargs: Arguments that are passed into the corresponding ``Strategy`` class.

    Returns:
        An initialized instance of the specified ``Strategy``.
    """
    assert isinstance(strategy_name, str), "`strategy_name` must be a string."
    match strategy_name.lower():
        case "default":
            return DefaultStrategy()

        case "fedasync" | "fed-async":
            from flox.strategies.impl.fedasync import FedAsync

            return FedAsync(**kwargs)

        case "fedavg" | "fed-avg":
            from flox.strategies.impl.fedavg import FedAvg

            return FedAvg(**kwargs)

        case "fedprox" | "fed-prox":
            from flox.strategies.impl.fedprox import FedProx

            return FedProx(**kwargs)

        case "fedsgd" | "fed-sgd":
            from flox.strategies.impl.fedsgd import FedSGD

            return FedSGD(**kwargs)

        case _:
            raise ValueError(f"Strategy '{strategy_name}' is not recognized.")


__all__ = [
    "Strategy",
    "ClientStrategy",
    "AggregatorStrategy",
    "WorkerStrategy",
    "TrainerStrategy",
    "load_strategy",
]
