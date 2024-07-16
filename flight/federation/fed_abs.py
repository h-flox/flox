import abc
import typing as t
from concurrent.futures import Future

from flight.strategies.aggr import AggrStrategy
from flight.strategies.coord import CoordStrategy
from flight.strategies.trainer import TrainerStrategy
from flight.strategies.worker import WorkerStrategy
from .jobs.types import Result, TrainJob, TrainJobArgs
from .jobs.work import default_training_job
from .topologies.node import Node
from .topologies.topo import Topology
from ..learning.datasets import DataLoadable

if t.TYPE_CHECKING:
    from .fed_sync import Engine

    Strategy: t.TypeAlias = t.Any
    Module: t.TypeAlias = t.Any
    Record: t.TypeAlias = dict[str, t.Any]


class Federation(abc.ABC):
    topology: Topology
    strategy: Strategy
    data: DataLoadable
    work_fn: TrainJob
    engine: Engine

    def __init__(
        self,
        topology: Topology,
        strategy: Strategy,
    ) -> None:
        self.topology = topology
        self.strategy = strategy
        self.work_fn = default_training_job

    @abc.abstractmethod
    def start(self, rounds: int) -> tuple[Module, list[Record]]:
        """Starts the federation.

        Returns:
            A tuple that contains the following items, (i) the trained global model hosted on the
            coordinator and (ii) the results from training during the federation.
        """

    @abc.abstractmethod
    def start_coordinator_task(
        self,
        node: Node,
    ) -> Future[Result]:
        """
        Prepare and submit the job for the coordinator.

        It will also submit the appropriate job to all the required children nodes. This is based on
        the selected children. Given the coordinator, $C$, and the set of selected worker nodes $W$,
        any node that falls on a path between $C$ and every worker $w \\in W$ will be have jobs sent
        to them.

        Args:
            node (Node): The Coordinator node.

        Returns:
            The aggregated results for the entire round.
        """

    @abc.abstractmethod
    def start_aggregator_task(
        self,
        node: Node,
        selected_children: t.Sequence[Node],
    ) -> Future[Result]:
        """
        Prepare and submit the job to the selected aggregator.

        It will also submit the appropriate job to the *selected* children of the aggregator `node`.

        Args:
            node (Node): The Aggregator node to run the aggregation function on.
            selected_children (t.Sequence[Node]): The children nodes of the

        Returns:
            The aggregated result.
        """

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

    def start_worker_task(self, node: Node, parent: Node) -> Future[Result]:
        """
        Prepares the arguments for the worker function and submits the function using
        the provided control plane via the given `Engine`.

        Args:
            node:
            parent:

        Returns:

        """
        args = TrainJobArgs(
            node=node,
            parent=parent,
            model=None,
            data=self.data,
            worker_strategy=self.worker_strategy,
            trainer_strategy=self.trainer_strategy,
        )
        args = self.engine.transfer(args)
        return self.engine(self.work_fn, args)

    def _resolve_node(self, node: Node | None) -> Node:
        """
        Resolves an ambiguous argument. Specifically, given a `Node` or a `None` value, return a `Node.

        Args:
            node (Node | None): Either a `Node` instance or `None`. If the value is `None`, then the
                Coordinator node is returned.

        Returns:
            The resolved `Node`.
        """
        if node is None:
            node = self.topology.coordinator
        if not isinstance(node, Node):
            raise ValueError(
                "Federation._resolve_node() failed to resolve the arg `node` to a `Node` instance. "
                "Must either be a `Node` instance or `None` (only if intended to be the Coordinator node)."
            )

        return node
