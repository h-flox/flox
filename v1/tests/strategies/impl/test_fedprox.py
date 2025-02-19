from v1.flight import (
    AggrStrategy,
    CoordStrategy,
    TrainerStrategy,
    WorkerStrategy,
)
from v1.flight.strategies.impl.fedavg import FedAvgWorker
from v1.flight.strategies.impl.fedprox import FedProx, FedProxTrainer
from v1.flight.strategies.impl.fedsgd import FedSGDAggr, FedSGDCoord


class TestValidFedProx:
    def test_fedprox_class_hierarchy(self):
        """Test that the associated node strategy types follow the correct protocols."""
        fedprox = FedProx(0.3, 1, False, False)
        assert isinstance(fedprox.aggr_strategy, (AggrStrategy, FedSGDAggr))
        assert isinstance(fedprox.coord_strategy, (CoordStrategy, FedSGDCoord))
        assert isinstance(fedprox.trainer_strategy, (TrainerStrategy, FedProxTrainer))
        assert isinstance(fedprox.worker_strategy, (WorkerStrategy, FedAvgWorker))
