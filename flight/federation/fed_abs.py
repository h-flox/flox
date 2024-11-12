from __future__ import annotations

import abc
import copy
import typing as t
from concurrent.futures import Future

from flight.strategies.aggr import AggrStrategy
from flight.strategies.coord import CoordStrategy
from flight.strategies.trainer import TrainerStrategy
from flight.strategies.worker import WorkerStrategy

from .jobs.types import Result, TrainJob, TrainJobArgs
from .jobs.work import default_training_job
from .topologies.node import Node, WorkerState

if t.TYPE_CHECKING:
    from ..engine import Engine
    from ..learning.base import AbstractDataModule, AbstractModule
    from ..strategies import Strategy
    from ..types import Record
    from .topologies.topo import Topology


def setup_work_job(fn: TrainJob | None) -> TrainJob:
    return default_training_job  # TODO
    match fn:
        case TrainJob():  # TODO: Currently illegal, `TrainJob` is not a type
            return default_training_job  # TODO
        case None:
            return default_training_job
        case _:
            raise ValueError(
                "Invalid value for work function provided. "
                "Must be either a `TrainJob` or `None`."
            )


class Federation(abc.ABC):
    topology: Topology
    strategy: Strategy
    data: AbstractDataModule
    work_fn: TrainJob
    engine: Engine
    global_model: AbstractModule
    work_fn: TrainJob

    def __init__(
        self,
        topology: Topology,
        strategy: Strategy,
        work_fn: TrainJob | None = None,
    ) -> None:
        self.topology = topology
        self.strategy = strategy
        self.work_fn = setup_work_job(work_fn)

    ####################################################################################

    @abc.abstractmethod
    def start(self, rounds: int) -> tuple[AbstractDataModule, list[Record]]:
        """
        Starts the federation.

        Args:
            rounds (int): The number of rounds to run the federation.

        Returns:
            A tuple that contains the following items:

                1. the trained global model hosted on the coordinator
                2. the results from training during the federation.
        """

    @abc.abstractmethod
    def coordinator_task(
        self,
        node: Node,
    ) -> Future[Result]:
        """
        Prepare and submit the job for the coordinator.

        It will also submit the appropriate job to all the required children nodes.
        This is based on the selected children. Given the coordinator, $C$, and the
        set of selected worker nodes $W$, any node that falls on a path between $C$
        and every worker $w \\in W$ will have jobs sent to them.

        Args:
            node (Node): The Coordinator node.

        Returns:
            The aggregated results for the entire round.
        """

    @abc.abstractmethod
    def aggregator_task(
        self,
        node: Node,
        selected_children: t.Sequence[Node],
    ) -> Future[Result]:
        """
        Prepare and submit the job to the selected aggregator.

        It will also submit the appropriate job to the *selected* children of
        the aggregator `node`.

        Args:
            node (Node): The Aggregator node to run the aggregation function on.
            selected_children (t.Sequence[Node]): The children nodes that will
                participate in federation for this round and perform local training.

        Returns:
            The aggregated result.
        """

    ####################################################################################

    @property
    def coord_strategy(self) -> CoordStrategy:
        """Convenience alias that returns the federation's `CoordStrategy`."""
        return self.strategy.coord_strategy

    @property
    def aggr_strategy(self) -> AggrStrategy:
        """Convenience alias that returns the federation's `AggrStrategy`."""
        return self.strategy.aggr_strategy

    @property
    def worker_strategy(self) -> WorkerStrategy:
        """Convenience alias that returns the federation's `WorkerStrategy`."""
        return self.strategy.worker_strategy

    @property
    def trainer_strategy(self) -> TrainerStrategy:
        """Convenience alias that returns the federation's `TrainerStrategy`."""
        return self.strategy.trainer_strategy

    ####################################################################################

    def worker_task(self, node: Node, parent: Node) -> Future[Result]:
        """
        Prepares the arguments for the worker function and submits the function using
        the provided controllers plane via the given `Engine`.

        Args:
            node (Node): The worker node.
            parent (Node): The worker node's parent.

        Returns:
            The future of the worker task.
        """
        state = WorkerState(node.idx)
        args = TrainJobArgs(
            node=node,
            parent=parent,
            node_state=state,
            model=copy.deepcopy(self.global_model),
            data=self.data,
            worker_strategy=self.worker_strategy,
            trainer_strategy=self.trainer_strategy,
        )
        args = self.engine.transfer(args)
        return self.engine.submit(self.work_fn, args=args)

    def _resolve_node(self, node: Node | None) -> Node:
        """
        Resolves an ambiguous argument. Specifically, given a `Node` or a `None` value,
        return a `Node`.

        Args:
            node (Node | None): Either a `Node` instance or `None`. If the value is
                `None`, then the Coordinator node is returned.

        Returns:
            The resolved `Node`.
        """
        if node is None:
            node = self.topology.coordinator

        if not isinstance(node, Node):
            raise ValueError(
                f"Federation._resolve_node() failed to resolve the arg `{node=}` to a "
                f"`Node` instance. Must either be a `Node` instance or `None` (only "
                f"if intended to be the Coordinator node)."
            )

        return node
