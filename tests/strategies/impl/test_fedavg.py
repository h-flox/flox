import typing as t

import torch

from flight.learning.types import Params
from flight.strategies import (
    AggrStrategy,
    CoordStrategy,
    TrainerStrategy,
    WorkerStrategy,
)
from flight.strategies.base import DefaultTrainerStrategy
from flight.strategies.impl.fedavg import FedAvg, FedAvgAggr, FedAvgWorker
from flight.strategies.impl.fedsgd import FedSGDCoord

if t.TYPE_CHECKING:
    NodeState: t.TypeAlias = t.Any


class TestValidFedAvg:
    def test_fedavg_class_hierarchy(self):
        """Test that the associated node strategy types follow the correct protocols."""
        fedavg = FedAvg()

        assert isinstance(fedavg.aggr_strategy, (AggrStrategy, FedAvgAggr))
        assert isinstance(fedavg.coord_strategy, (CoordStrategy, FedSGDCoord))
        assert isinstance(
            fedavg.trainer_strategy, (TrainerStrategy, DefaultTrainerStrategy)
        )
        assert isinstance(fedavg.worker_strategy, (WorkerStrategy, FedAvgWorker))

    def test_fedavg_aggr(self):
        """Tests the usability of the aggregator strategy for 'FedAvg'"""
        fedavg = FedAvg()
        aggregatorStrat: AggrStrategy = fedavg.aggr_strategy
        node_state: NodeState = {}
        child_states = {
            1: {"num_data_samples": 1, "other_data": "foo"},
            2: {"num_data_samples": 1, "other_data": "foo"},
        }
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

        aggregated = aggregatorStrat.aggregate_params(
            node_state, child_states, children_state_dicts_pt
        )

        assert isinstance(aggregated, dict)

        expected_avg = {
            "train/loss": 2.7,
            "train/acc": 1.3,
        }

        epsilon = 1e-6
        for key, value in aggregated.items():
            expected = expected_avg[key]
            assert abs(expected - value.item()) < epsilon

    def test_fedavg_worker(self):
        """Tests the usability of the worker strategy for 'FedAvg'"""
        fedavg = FedAvg()

        workerStrat: WorkerStrategy = fedavg.worker_strategy

        nodestate_before: NodeState = {"State:": "Training preperation"}
        data_before: Params = {
            "train/loss1": torch.tensor(0.35, dtype=torch.float32),
            "train/loss2": torch.tensor(0.5, dtype=torch.float32),
            "train/loss3": torch.tensor(0.23, dtype=torch.float32),
        }

        nodestate_after, data_after = workerStrat.before_training(
            nodestate_before, data_before
        )

        assert nodestate_after["num_data_samples"] == len(data_before)
        assert data_before == data_after
