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
    """Helper function used by aggregator nodes for averaging the passed node state dictionary.

    Args:
        state_dicts (Mapping[NodeID, Params]): A dictionary object mapping nodes to their respective states.
        weights (Mapping[NodeID, float] | None, optional): Optional dictionary that maps each node to its contribution factor. Defaults to None.

    Returns:
        Params: The averaged parameters.
    """
    num_nodes = len(state_dicts)

    if weights is not None:
        weight_sum = numpy.sum(list(weights.values()))
    else:
        weight_sum = None

    with torch.no_grad():
        avg_weights = {}
        for node, state_dict in state_dicts.items():
            if weights is not None:
                w = weights[node] / weight_sum
            else:
                w = 1 / num_nodes
            for name, value in state_dict.items():
                value = w * torch.clone(value)
                if name not in avg_weights:
                    avg_weights[name] = value
                else:
                    avg_weights[name] += value

    return avg_weights
