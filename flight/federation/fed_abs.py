import abc
import typing as t

from flight.strategies.aggr import AggrStrategy
from flight.strategies.coord import CoordStrategy
from flight.strategies.trainer import TrainerStrategy
from flight.strategies.worker import WorkerStrategy

from .topologies.node import Node
from .topologies.topo import Topology

if t.TYPE_CHECKING:
    Strategy: t.TypeAlias = t.Any
    Module: t.TypeAlias = t.Any
    Record: t.TypeAlias = dict[str, t.Any]


class Federation(abc.ABC):
    topology: Topology
    strategy: Strategy

    def __init__(
        self,
        topology: Topology,
        strategy: Strategy,
    ) -> None:
        self.topology = topology
        self.strategy = strategy

    @abc.abstractmethod
    def start(self, rounds: int) -> tuple[Module, list[Record]]:
        """Starts the federation.

        Returns:
            A tuple that contains the following items, (i) the trained global model hosted on the
            coordinator and (ii) the results from training during the federation.
        """
        pass

    @property
    def coord_strategy(self) -> CoordStrategy:
        return self.strategy.coord_strategy

    @property
    def aggr_strategy(self) -> AggrStrategy:
        return self.strategy.aggr_strategy

    @property
    def worker_strategy(self) -> WorkerStrategy:
        return self.strategy.worker_strategy

    @property
    def trainer_strategy(self) -> TrainerStrategy:
        return self.strategy.trainer_strategy

    def _resolve_node(self, node: Node | None):
        if node is None:
            node = self.topology.coord
        if not isinstance(node, Node):
            raise ValueError("Federation._resolve_node() failed to resolve the node.")
