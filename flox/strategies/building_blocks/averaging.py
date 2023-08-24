import torch

from flox.typing import StateDict


def average_state_dicts(state_dicts: list[StateDict]) -> StateDict:
    with torch.no_grad():
        avg_weights = {}
        nk = 1 / len(state_dicts)
        for sd in state_dicts:
            for name, value in sd.items():
                value = nk * torch.clone(value)
                if name not in avg_weights:
                    avg_weights[name] = value
                else:
                    avg_weights[name] += value

    return avg_weights
