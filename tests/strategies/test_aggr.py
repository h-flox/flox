import typing as t

import pytest
import torch

from flight.federation.topologies.node import NodeState
from flight.strategies import AggrStrategy
from flight.strategies.base import DefaultAggrStrategy
from flight.strategies.commons import average_state_dicts

W, B = "weight", "bias"


def test_instance():
    """Test that the associated node strategy type follows the correct protocols."""
    default_aggr = DefaultAggrStrategy()

    assert isinstance(default_aggr, AggrStrategy)


def test_aggr_aggregate_params():
    """Tests usability for the 'aggregate_params' function on two children."""
    default_aggr = DefaultAggrStrategy()

    state: NodeState = "foo"
    children = {1: "foo1", 2: "foo2"}

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

    avg = default_aggr.aggregate_params(state, children, children_state_dicts_pt)

    assert isinstance(avg, dict)

    expected_avg = {
        "train/loss": 2.7,
        "train/acc": 1.3,
    }

    epsilon = 1e-6
    for key, value in avg.items():
        expected = expected_avg[key]
        assert abs(expected - value.item()) < epsilon


class TestAveraging:
    @pytest.fixture
    def params(self) -> dict[int, dict[str, torch.Tensor]]:
        return {
            0: {W: torch.tensor([10.0]), B: torch.tensor([5.0])},
            1: {W: torch.tensor([15.0]), B: torch.tensor([2.5])},
        }

    @pytest.fixture
    def weights(self) -> dict[t, float]:
        return {
            0: 0.0,
            1: 1.0,
        }

    def test_correct_averaging(self, params, weights):
        avg_params = average_state_dicts(params)
        assert avg_params[W].item() == (params[0][W] + params[1][W]) / 2
        assert avg_params[B].item() == (params[0][B] + params[1][B]) / 2

        avg_params = average_state_dicts(params, weights)
        assert avg_params[W] == params[1][W]
        assert avg_params[B] == params[1][B]
