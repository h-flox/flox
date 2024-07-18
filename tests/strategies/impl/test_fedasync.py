import typing as t

import pytest
import torch

from flight.strategies import (
    AggrStrategy,
    CoordStrategy,
    DefaultStrategy,
    NodeState,
    TrainerStrategy,
    WorkerStrategy,
)
from flight.strategies.base import (
    DefaultCoordStrategy,
    DefaultTrainerStrategy,
    DefaultWorkerStrategy,
)
from flight.strategies.impl.fedasync import FedAsync, FedAsyncAggr


class TestValidFedAsync:
    def test_class_hierarchy(self):
        fedasync = FedAsync(0.5)

        assert (
            isinstance(fedasync.aggr_strategy, (AggrStrategy, FedAsyncAggr))
            and isinstance(
                fedasync.coord_strategy, (CoordStrategy, DefaultCoordStrategy)
            )
            and isinstance(
                fedasync.trainer_strategy, (TrainerStrategy, DefaultTrainerStrategy)
            )
            and isinstance(
                fedasync.worker_strategy, (WorkerStrategy, DefaultWorkerStrategy)
            )
        )

    def test_fedasync_aggr(self):
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
        with pytest.raises(AssertionError):
            fedasync = FedAsync(alpha=1.1)
