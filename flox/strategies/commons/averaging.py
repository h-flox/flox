import numpy as np
import torch

from flox.flock import FlockNodeID
from flox.typing import StateDict


def average_state_dicts(
    state_dicts: dict[FlockNodeID, StateDict],
    weights: dict[FlockNodeID, float] | None = None,
) -> StateDict:
    """Averages the parameters given by ``global_module.state_dict()`` from a set of ``FlockNodes``.

    Args:
        state_dicts (dict[FlockNodeID, StateDict]): The global_module state dicts of each FlockNode to average.
        weights (dict[FlockNodeID, float] | None): The weights for each ``FlockNode`` used do weighted averaging. If
            no weights are provided (i.e., `weights=None`), then standard averaging is done.

    Returns:
        Averaged weights as a ``StateDict``.
    """
    num_nodes = len(state_dicts)
    weight_sum = None if weights is None else np.sum(list(weights.values()))

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
