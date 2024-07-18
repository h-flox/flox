from __future__ import annotations

import functools
import typing as t

import pydantic as pyd

from flight.strategies.aggr import AggrStrategy
from flight.strategies.commons.averaging import average_state_dicts
from flight.strategies.coord import CoordStrategy
from flight.strategies.trainer import TrainerStrategy
from flight.strategies.worker import WorkerStrategy

StrategyType: t.TypeAlias = (
    WorkerStrategy | AggrStrategy | CoordStrategy | TrainerStrategy
)

if t.TYPE_CHECKING:
    import torch
    from numpy.random import Generator

    from flight.federation.jobs.result import Result
    from flight.federation.topologies.node import Node, NodeID
    from flight.strategies import Loss, NodeState, Params


class DefaultCoordStrategy:
    """Default implementation of the strategy for a coordinator."""

    def select_workers(
        self, state: NodeState, workers: t.Iterable[Node], rng: Generator
    ) -> t.Sequence[Node]:
        """Method used for the selection of workers.

        Args:
            state (NodeState): The state of the coordinator node.
            workers (t.Iterable[Node]): Iterable object containing all of the worker nodes.
            rng (Generator): RNG object used for randomness.

        Returns:
            t.Sequence[Node]: The selected workers.
        """
        return list(workers)


class DefaultAggrStrategy:
    """Default implementation of the strategy for an aggregator."""

    def start_round(self):
        pass

    def aggregate_params(
        self,
        state: NodeState,
        children_states: t.Mapping[NodeID, NodeState],
        children_state_dicts: t.Mapping[NodeID, Params],
        **kwargs,
    ) -> Params:
        """Callback that handles the model parameter aggregation step.

        Args:
            state (NodeState): The state of the current aggregator node.
            children_states (t.Mapping[NodeID, NodeState]): A mapping of the current aggregator node's children and their respective states.
            children_state_dicts (t.Mapping[NodeID, Parmas]): The model parameters of the models to each respective child node.
            **kwargs: Keyword arguments provided by users.

        Returns:
            Params: The aggregated values.
        """
        return average_state_dicts(children_state_dicts, weights=None)

    def end_round(self):
        pass


class DefaultWorkerStrategy:
    """Default implementation of the strategy for a worker"""

    def start_work(self, state: NodeState) -> NodeState:
        """Callback to be ran and the start of the current worker nodes work.

        Args:
            state (NodeState): The state of the current worker node.

        Returns:
            NodeState: The state of the current worker node at the end of the callback.
        """
        return state

    def before_training(
        self, state: NodeState, data: Params
    ) -> tuple[NodeState, Params]:
        """Callback to be ran before training.

        Args:
            state (NodeState): The state of the current worker node.
            data (Params): The data associated with the current worker node.

        Returns:
            tuple[NodeState, Params]: A tuple containing the state and data of the worker node at the end of the callback.
        """
        return state, data

    def after_training(
        self, state: NodeState, optimizer: torch.optim.Optimizer
    ) -> NodeState:
        """Callback to be ran after training.

        Args:
            state (NodeState): The state of the current worker node.
            optimizer (torch.optim.Optimizer): The PyTorch optimier to be used.

        Returns:
            NodeState: The state of the worker node at the end of the callback.
        """
        return state

    def end_work(self, result: Result) -> Result:
        """Callback to be ran at the end of the work.

        Args:
            result (Result): A Result object used to represent the result of the local training on the current worker node.

        Returns:
            Result: The result of the worker nodes local training.
        """
        return result


class DefaultTrainerStrategy:
    """Default implementation of a strategy for the trainer."""

    def before_backprop(self, state: NodeState, loss: Loss) -> Loss:
        """Callback to run before backpropagation.

        Args:
            state (NodeState): State of the current node.
            loss (Loss): The calculated loss

        Returns:
            The loss at the end of the callback
        """
        return loss

    def after_backprop(self, state: NodeState, loss: Loss) -> Loss:
        """Callback to run after backpropagation.

        Args:
            state (NodeState): State of the current node.
            loss (Loss): The calculated loss

        Returns:
            The loss at the end of the callback
        """
        return loss


# TODO: Remove config when all type definitions have been resolved
@pyd.dataclasses.dataclass(
    frozen=True, repr=False, config={"arbitrary_types_allowed": True}
)
class Strategy:
    """
    A 'Strategy' implementation is comprised of the four different type of implementations of strategies
    to be used on the respective node types throughout the training process.
    """

    """Implementation of the specific callbacks for the coordinator node."""
    coord_strategy: CoordStrategy = pyd.Field()
    """Implementation of the specific callbacks for the aggregator node(s)."""
    aggr_strategy: AggrStrategy = pyd.Field()
    """Implementation of the specific callbacks for the worker node(s)."""
    worker_strategy: WorkerStrategy = pyd.Field()
    """Implementation of callbacks specific to the execution of the training loop on the worker node(s)."""
    trainer_strategy: TrainerStrategy = pyd.Field()

    def __iter__(self) -> t.Iterator[tuple[str, StrategyType]]:
        yield from (
            ("coord_strategy", self.coord_strategy),
            ("aggr_strategy", self.aggr_strategy),
            ("worker_strategy", self.worker_strategy),
            ("trainer_strategy", self.trainer_strategy),
        )

    def __repr__(self) -> str:
        return str(self)

    @functools.cached_property
    def _description(self) -> str:
        """A utility function for generating the string for `__str__`.

        This is written to avoid the `mypy` issue:
            "Signature of '__str__' incompatible with supertype 'object'".

        Returns:
            The string representation of the a Strategy instance.
        """
        name = self.__class__.__name__
        inner = ", ".join(
            [
                f"{strategy_key}={strategy_value.__class__.__name__}"
                for (strategy_key, strategy_value) in iter(self)
                if strategy_value is not None
            ]
        )
        return f"{name}({inner})"

    def __str__(self) -> str:
        return self._description


class DefaultStrategy(Strategy):
    """Implementation of a strategy that uses the default strategy types for each node type."""

    def __init__(self) -> None:
        super().__init__(
            coord_strategy=DefaultCoordStrategy(),
            aggr_strategy=DefaultAggrStrategy(),
            worker_strategy=DefaultWorkerStrategy(),
            trainer_strategy=DefaultTrainerStrategy(),
        )
