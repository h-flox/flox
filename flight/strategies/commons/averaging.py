from __future__ import annotations

import typing as t

import numpy
import torch

if t.TYPE_CHECKING:
    from collections.abc import Mapping

    NodeID: t.TypeAlias = t.Any
    Params: t.TypeAlias = t.Any


def average_state_dicts(
    state_dicts: Mapping[NodeID, Params], weights: Mapping[NodeID, float] | None = None
) -> Params:
    num_nodes = len(state_dicts)
    weight_sum = None if weights is None else numpy.sum(list(weights.values()))

    with torch.no_grad():
        avg_weights = {}
        for node, state_dict in state_dicts.items():
            w = 1 / num_nodes if weights is None else weights[node] / weight_sum
            for name, value in state_dict.items():
                value = w * torch.clone(value)
                if name not in avg_weights:
                    avg_weights[name] = value
                else:
                    avg_weights[name] += value

    return avg_weights
