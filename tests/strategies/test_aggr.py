from flight.strategies import AggrStrategy
from flight.strategies.base import DefaultAggrStrategy

import tensorflow as tf
import torch


def test_instance():
    default_aggr = DefaultAggrStrategy()

    assert isinstance(default_aggr, AggrStrategy)


def test_aggr_aggregate_params():
    default_aggr = DefaultAggrStrategy()

    state = "foo"
    children = {1: "foo1", 2: "foo2"}

    children_state_dicts = {
        1: {
            "train/loss": tf.convert_to_tensor(2.3, dtype=tf.float32),
            "train/acc": tf.convert_to_tensor(1.2, dtype=tf.float32),
        },
        2: {
            "train/loss": tf.convert_to_tensor(3.1, dtype=tf.float32),
            "train/acc": tf.convert_to_tensor(1.4, dtype=tf.float32),
        },
    }

    children_state_dicts_pt = {
        key: {
            sub_key: torch.tensor(value.numpy()) for sub_key, value in sub_dict.items()
        }
        for key, sub_dict in children_state_dicts.items()
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

        assert abs(expected - value) < epsilon
