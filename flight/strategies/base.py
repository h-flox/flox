from __future__ import annotations

import dataclasses
import functools
import typing as t

from flight.learning import AbstractModule
from flight.strategies.aggr import AggrStrategy
from flight.strategies.commons import average_state_dicts
from flight.strategies.coord import CoordStrategy
from flight.strategies.trainer import TrainerStrategy
from flight.strategies.worker import WorkerStrategy

StrategyTypes: t.TypeAlias = (
    WorkerStrategy | AggrStrategy | CoordStrategy | TrainerStrategy
)

if t.TYPE_CHECKING:
    from flight.federation.topologies.node import AggrState, NodeID, NodeState
    from flight.learning import NpParams, Params


class DefaultCoordStrategy(CoordStrategy):
    """Default implementation of the strategy for a coordinator."""

    # def select_workers(
    #     self, state: NodeState, workers: t.Iterable[Node], rng: Generator
    # ) -> t.Sequence[Node]:
    #     """Method used for the selection of workers.
    #
    #     Args:
    #         state (NodeState): The state of the coordinator node.
    #         workers (t.Iterable[Node]): Iterable object containing all the worker
    #           nodes.
    #         rng (Generator): RNG object used for randomness.
    #
    #     Returns:
    #         Worker nodes selected to participate in a federation round.
    #     """
    #     return list(workers)


class DefaultAggrStrategy(AggrStrategy):
    """Default implementation of the strategy for an aggregator."""

    # def start_round(self):
    #     """
    #     Callback to run at the start of a round.
    #     """
    #     pass

    def aggregate_params(
        self,
        state: AggrState,
        children_states: t.Mapping[NodeID, NodeState],
        children_modules: t.Mapping[NodeID, AbstractModule],
        **kwargs: dict[str, t.Any],
    ) -> Params:
        """Callback that handles the model parameter aggregation step.

        Args:
            state (AggrState): The state of the current aggregator node.
            children_states (t.Mapping[NodeID, NodeState]): A mapping of the current
                aggregator node's children and their respective states.
            children_modules (t.Mapping[NodeID, AbstractModule]): A mapping of the
                modules belonging to each respective child node.
            **kwargs: Keyword arguments provided by users.

        Returns:
            Aggregated parameters.
        """
        children_params: dict[NodeID, NpParams] = {}
        for idx in children_states:
            children_params[idx] = children_modules[idx].get_params(to_numpy=True)

        return average_state_dicts(children_params, weights=None)

    # def end_round(self):
    #     """
    #     Callback to run at the end of a round.
    #     """
    #     pass


class DefaultWorkerStrategy(WorkerStrategy):
    """Default implementation of the strategy for a worker"""

    # def start_work(self, state: WorkerState) -> WorkerState:
    #     """Callback that is run at the start of the current worker node's work.
    #
    #     Args:
    #         state (WorkerState): The state of the current worker node.
    #
    #     Returns:
    #         WorkerState: The state of the current worker node at the end
    #             of the callback.
    #     """
    #     return state
    #
    # def before_training(
    #     self,
    #     state: WorkerState,
    #     data: AbstractDataModule,  # TODO: Refactor later?
    # ) -> tuple[WorkerState, AbstractDataModule]:
    #     """Callback that is run before training.
    #
    #     Args:
    #         state (WorkerState): The state of the current worker node.
    #         data (AbstractDataModule): The data associated with the current worker
    #           node.
    #
    #     Returns:
    #         tuple[NodeState, Params]: A tuple containing the state and data of the
    #             worker node at the end of the callback.
    #     """
    #     return state, data
    #
    # def after_training(
    #     self,
    #     state: WorkerState,
    #     optimizer: torch.optim.Optimizer,
    # ) -> WorkerState:
    #     """Callback that is run after training.
    #
    #     Args:
    #         state (WorkerState): The state of the current worker node.
    #         optimizer (torch.optim.Optimizer): The PyTorch optimizer to be used.
    #
    #     Returns:
    #         NodeState: The state of the worker node at the end of the callback.
    #     """
    #     return state
    #
    # def end_work(self, result: Result) -> Result:
    #     """Callback to be run at the end of the work.
    #
    #     Args:
    #         result (Result): A Result object used to represent the result of the local
    #             training on the current worker node.
    #
    #     Returns:
    #         Result: The result of the worker nodes local training.
    #     """
    #     return result


class DefaultTrainerStrategy(TrainerStrategy):
    """Default implementation of a strategy for the trainer."""

    # def hparams(
    #     self,
    #     node: Node | None = None,
    #     state: WorkerState | None = None,
    # ) -> dict[str, t.Any]:
    #     return {}
    #
    # def before_backprop(
    #     self,
    #     state: WorkerState,
    #     out: LocalStepOutput,
    # ) -> LocalStepOutput:
    #     """Callback to run before backpropagation.
    #
    #     Args:
    #         state (WorkerState): State of the current node.
    #         out (LocalStepOutput): The calculated loss
    #
    #     Returns:
    #         The loss at the end of the callback
    #     """
    #     return out
    #
    # def after_backprop(
    #     self,
    #     state: WorkerState,
    #     out: LocalStepOutput,
    # ) -> LocalStepOutput:
    #     """Callback to run after backpropagation.
    #
    #     Args:
    #         state (WorkerState): State of the current node.
    #         out (LocalStepOutput): The calculated loss
    #
    #     Returns:
    #         The loss at the end of the callback
    #     """
    #     return out


@dataclasses.dataclass(frozen=True, repr=False)
class Strategy:
    """
    A 'Strategy' implementation is made up of the four different type of
    implementations of strategies to be used on the respective node types throughout
    the training process.
    """

    coord_strategy: CoordStrategy
    """
    Implementation of the specific callbacks for the coordinator node.
    """

    aggr_strategy: AggrStrategy
    """
    Implementation of the specific callbacks for the aggregator node(s).
    """

    worker_strategy: WorkerStrategy
    """
    Implementation of the specific callbacks for the worker node(s).
    """

    trainer_strategy: TrainerStrategy
    """
    Implementation of callbacks specific to the execution of the training loop
    on the worker node(s).
    """

    def __iter__(self) -> t.Iterator[tuple[str, StrategyTypes]]:
        items: list[tuple[str, StrategyTypes]] = [
            ("coord_strategy", self.coord_strategy),
            ("aggr_strategy", self.aggr_strategy),
            ("worker_strategy", self.worker_strategy),
            ("trainer_strategy", self.trainer_strategy),
        ]
        yield from items

    def __str__(self) -> str:
        return self._description

    def __repr__(self) -> str:
        return str(self)

    @functools.cached_property
    def _description(self) -> str:
        """A utility function for generating the string for `__str__`.

        This is written to avoid the `mypy` issue:
            "Signature of '__str__' incompatible with supertype 'object'".

        Returns:
            The string representation of the Strategy instance.
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


class DefaultStrategy(Strategy):
    """
    Implementation of a strategy that uses the default strategy types for
    each node type.
    """

    def __init__(self) -> None:
        super().__init__(
            coord_strategy=DefaultCoordStrategy(),
            aggr_strategy=DefaultAggrStrategy(),
            worker_strategy=DefaultWorkerStrategy(),
            trainer_strategy=DefaultTrainerStrategy(),
        )
