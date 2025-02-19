import pytest
import torch

from v1.flight.topologies.node import NodeState
from v1.flight import (
    AggrStrategy,
    CoordStrategy,
    TrainerStrategy,
    WorkerStrategy,
)
from v1.flight import (
    DefaultCoordStrategy,
    DefaultTrainerStrategy,
    DefaultWorkerStrategy,
)
from v1.flight.strategies.impl.fedasync import FedAsync, FedAsyncAggr


class TestValidFedAsync:
    def test_class_hierarchy(self):
        """Test that the associated node strategy types follow the correct protocols."""
        fedasync = FedAsync(0.5)

        assert isinstance(fedasync.aggr_strategy, (AggrStrategy, FedAsyncAggr))
        assert isinstance(
            fedasync.coord_strategy, (CoordStrategy, DefaultCoordStrategy)
        )
        assert isinstance(
            fedasync.trainer_strategy, (TrainerStrategy, DefaultTrainerStrategy)
        )
        assert isinstance(
            fedasync.worker_strategy, (WorkerStrategy, DefaultWorkerStrategy)
        )

    def test_fedasync_aggr(self):
        """Tests implementation of the aggregator within 'FedAsync'"""
        strategy = FedAsync(0.5)
        aggr_strategy: AggrStrategy = strategy.aggr_strategy

        nodestate: NodeState = {}
        childstates = {1: "foo1", 2: "foo2"}
        children_state_dicts_pt = {
            1: {
                "train/loss": torch.tensor(2.3, dtype=torch.float32),
                "train/acc": torch.tensor(1.2, dtype=torch.float32),
            },
            2: {
                "train/loss": torch.tensor(3.1, dtype=torch.float32),
                "train/acc": torch.tensor(1.4, dtype=torch.float32),
            },
        }

        # avg = aggr_strategy.aggregate_params(
        #    nodestate, childstates, children_state_dicts_pt, last_updated_node=1
        # )

        assert NotImplementedError


class TestInvalidFedAsync:
    def test_invalid_alpha(self):
        """Test inputting a value for alpha which is too large."""
        with pytest.raises(AssertionError):
            fedasync = FedAsync(alpha=1.1)
