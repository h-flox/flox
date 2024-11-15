from __future__ import annotations

import typing as t

import numpy as np

if t.TYPE_CHECKING:
    from flight.federation.topologies.node import NodeID
    from flight.learning.types import NpParams


def average_state_dicts(
    state_dicts: t.Mapping[NodeID, NpParams],
    weights: t.Mapping[NodeID, float] | None = None,
) -> NpParams:
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
    for node, state_dict in state_dicts.items():
        w = node_weights[node]
        for name, value in state_dict.items():
            if name not in avg_weights:
                avg_weights[name] = w * np.copy(value)
            else:
                avg_weights[name] += w * np.copy(value)

    # with torch.no_grad():
    #     avg_weights = {}
    #     for node, state_dict in state_dicts.items():
    #         w = node_weights[node]
    #         for name, value in state_dict.items():
    #             match value:
    #                 # TODO: We need some abstraction for math operations across numpy
    #                 #       and tensors.
    #                 case torch.Tensor():
    #                     value = w * torch.clone(value)
    #                 case np.ndarray():
    #                     value = w * value
    #                 case _:
    #                     raise ValueError("Unsupported data type for parameter value.")
    #
    #             if name not in avg_weights:
    #                 avg_weights[name] = value
    #             else:
    #                 avg_weights[name] += value

    return avg_weights
