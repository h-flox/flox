from __future__ import annotations

import typing

import numpy
import torch

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from flox.topos import NodeID
    from flox.learn.typing import Params


def average_state_dicts(
    state_dicts: Mapping[NodeID, Params],
    weights: Mapping[NodeID, float] | None = None,
) -> Params:
    """Averages the parameters given by ``global_model.params()`` from a set of ``FlockNodes``.

    Args:
        state_dicts (dict[NodeID, Params]): The global_model state dicts of each FlockNode to average.
        weights (dict[NodeID, float] | None): The weights for each ``FlockNode`` used do weighted averaging. If
            no weights are provided (i.e., `weights=None`), then standard averaging is done.

    Returns:
        Averaged weights as a ``Params``.
    """
    num_nodes = len(state_dicts)
    weight_sum = None if weights is None else numpy.sum(list(weights.values()))

    with torch.no_grad():
        avg_weights = {}
        for node, state_dict in state_dicts.items():
            w = 1 / num_nodes if weights is None else weights[node] / weight_sum  # type: ignore
            for name, value in state_dict.items():
                value = w * torch.clone(value)
                if name not in avg_weights:
                    avg_weights[name] = value
                else:
                    avg_weights[name] += value

    return avg_weights
