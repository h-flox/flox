from __future__ import annotations

import collections as c
import typing as t

import numpy as np

from flight.learning.parameters import Params, parameters


def average_params(params: t.Collection[Params]) -> Params:
    r"""
    A simple averaging method for a list of parameters.

    This function takes a list of parameter dictionaries and computes the average
    via
    $$
    \omega_\text{avg} \triangleq \frac{1}{N} \sum_{i=1}^{N} \omega_i
    $$

    where $N$ is the number of parameter sets (i.e., `N = len(params)`) and $\omega_i$
    are the individual parameter sets.

    Args:
        params (list[Params]): A list of parameters to average.

    Returns:
        A new [`Params`][flight.learning.module.Params]
        object containing the averaged parameters.
    """
    keys = set(next(iter(params)).keys())
    avg_params: dict[str, t.Any] = c.OrderedDict()
    weight = 1 / len(params)

    for p in params:
        for key in p.keys():
            if key not in keys:
                raise ValueError(f"Parameter key {key} not found in all parameter sets")
            elif key in avg_params:
                avg_params[key] += p[key] * weight
            else:
                avg_params[key] = p[key] * weight

    return parameters(avg_params)


def weighted_average_params(
    params: t.Collection[Params],
    weights: t.Collection[int | float],
) -> Params:
    r"""
    A weighted averaging method for a list of parameters.

    This function takes a list of parameter dictionaries and computes the average
    via
    $$
    \omega_\text{avg} \triangleq \sum_{i=1}^{N} c_{i} \cdot \omega_i
    $$

    where $c_{i}$ is the weighting coefficient for parameter set $i$,
    $N$ is the number of parameter sets (i.e., `N = len(params)`), and $\omega_i$
    are the individual parameter sets.

    Args:
        params (list[Params]): A list of parameters to average.
        weights (t.Iterable[int | float]): A list of weights for each parameter set.
            The weights must sum to 1.

    Returns:
        A new [`Params`][flight.learning.module.Params]
        object containing the averaged parameters.

    Notes:
        The values of `weights` should sum to 1. If they do not, they will be
        normalized to sum to 1 via `weights = weights / weights.sum()`.
    """
    if len(params) != len(weights):
        raise ValueError(
            f"Number of parameters ({len(params)}) must match number of "
            f"weights ({len(weights)})."
        )

    weights_np = np.array(weights, dtype=float)
    if weights_np.sum() != 1:
        weights_np = weights_np / weights_np.sum()

    keys = set(next(iter(params)).keys())
    avg_params: dict[str, t.Any] = c.OrderedDict()

    for i, p in enumerate(params):
        for key in p.keys():
            if key not in keys:
                raise ValueError(f"Parameter key {key} not found in all parameter sets")
            elif key in avg_params:
                avg_params[key] += p[key] * weights_np[i]
            else:
                avg_params[key] = p[key] * weights_np[i]

    return parameters(avg_params)
