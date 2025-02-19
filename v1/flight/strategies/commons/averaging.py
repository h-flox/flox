from __future__ import annotations

import typing as t

import numpy as np

from v1.flight.learning.params import Params

if t.TYPE_CHECKING:
    from v1.flight.topologies.node import NodeID


def average_state_dicts(
    state_dicts: t.Mapping[NodeID, Params],
    weights: t.Mapping[NodeID, float] | None = None,
) -> Params:
    """
    Common implementation for averaging model parameters.

    This helper function supports weighted and unweighted averaging. The latter is
    done when `weights` is set to `None`.

    Args:
        state_dicts (t.Mapping[NodeID, Params]): A dictionary object mapping nodes to
            their respective states.
        weights (t.Mapping[NodeID, float] | None, optional): Optional dictionary that
            maps each node to its contribution factor. Defaults to `None`.

    Returns:
        The averaged parameters.
    """
    num_nodes = len(state_dicts)

    if weights is None:
        node_weights = {node: 1 / num_nodes for node in state_dicts}
    else:
        weight_sum = sum(list(weights.values()))
        node_weights = {node: weights[node] / weight_sum for node in weights}

    avg_weights = {}
    for node, node_params in state_dicts.items():
        node_params = node_params.numpy()
        w = node_weights[node]
        for name, value in node_params.items():
            if name not in avg_weights:
                avg_weights[name] = w * np.copy(value)
            else:
                avg_weights[name] += w * np.copy(value)

    return Params(avg_weights)
