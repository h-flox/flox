import typing as t

import torch

from flight.strategies import AggrStrategy
from flight.strategies.base import DefaultAggrStrategy

if t.TYPE_CHECKING:
    NodeState: t.TypeAlias = t.Any


def test_instance():
    default_aggr = DefaultAggrStrategy()

    assert isinstance(default_aggr, AggrStrategy)


def test_aggr_aggregate_params():
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
