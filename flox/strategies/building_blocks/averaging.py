import numpy as np
import torch

from typing import Optional

from flox.flock import FlockNodeID
from flox.typing import StateDict


def average_state_dicts(
    state_dicts: dict[FlockNodeID, StateDict],
    weights: Optional[dict[FlockNodeID, float]] = None,
):
    num_nodes = len(state_dicts)
    weight_sum = None if weights is None else np.sum(list(weights.values()))

    with torch.no_grad():
        avg_weights = {}
        for node, state_dict in state_dicts.items():
            if weights is None:
                w = 1 / num_nodes
            else:
                w = weights[node] / weight_sum
            for name, value in state_dict.items():
                value = w * torch.clone(value)
                if name not in avg_weights:
                    avg_weights[name] = value
                else:
                    avg_weights[name] += value

    return avg_weights
