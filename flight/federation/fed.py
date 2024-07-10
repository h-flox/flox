import abc
import typing as t

from flight.strategies.aggr import AggrStrategy
from flight.strategies.coord import CoordStrategy
from flight.strategies.trainer import TrainerStrategy
from flight.strategies.worker import WorkerStrategy

if t.TYPE_CHECKING:
    Strategy: t.TypeAlias = t.Any
    Module: t.TypeAlias = t.Any

    Record: t.TypeAlias = dict[str, t.Any]


class Federation(abc.ABC):
    strategy: Strategy

    @abc.abstractmethod
    def start(self) -> tuple[Module, list[Record]]:
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