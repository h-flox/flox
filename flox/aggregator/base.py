from __future__ import annotations

import torch

from typing import Any, NewType, Optional, TypeVar

E = TypeVar("E")
StateDict = NewType("StateDict", dict[str, torch.Tensor])


# TODO: At some point, we'll need to have a version that supports returning
#       `dict[E, StateDict]` for personalized FL.
class AggregatorFn:
    @staticmethod
    def __call__(
        module: torch.nn.Module,
        state_dicts: dict[E, StateDict],
        extra_info: Optional[dict[E, Any] | dict[E, dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> StateDict:
        raise NotImplementedError()


class SimpleAvg(AggregatorFn):
    @staticmethod
    def __call__(
        module: torch.nn.Module,
        state_dicts: dict[E, StateDict],
        extra_info: Optional[
            dict[E, Any] | dict[E, dict[str, Any]]
        ] = None,  # NOTE: What goes here into this is Logic-specific...
        *args,
        **kwargs,
    ) -> StateDict:
        with torch.no_grad():
            avg_weights = {}
            nk = 1 / len(state_dicts)
            for sd in state_dicts.values():
                for name, value in sd.items():
                    value = nk * torch.clone(value)
                    if name not in avg_weights:
                        avg_weights[name] = value
                    else:
                        avg_weights[name] += value

        return StateDict(avg_weights)
