from __future__ import annotations

import functools
import typing as t
from collections import OrderedDict
from enum import Enum, auto

import numpy as np
import torch

NpParams: t.TypeAlias = dict[str, np.ndarray]
"""
Type alias for model parameters as a mapping where the keys are strings and
the values are Numpy `ndarray`s.
"""

TorchParams: t.TypeAlias = dict[str, torch.Tensor]
"""
Type alias for model parameters as a mapping where the keys are strings and
the values are parameters as PyTorch `Tensor`s.
"""


class UnsupportedParameterKindError(ValueError):
    """
    An Exception raised when an unsupported parameter kind is detected.
    """

    def __init__(self, message: str | None = None, *args):
        if message is None:
            message = (
                "The parameter kind is unknown or unsupported. "
                "Please refer to the docs."
            )
        super().__init__(message, *args)


class InconsistentParamValuesError(ValueError):
    """
    An Exception raised when the parameter value kinds are inconsistent.
    """

    def __init__(self, message: str | None = None, *args):
        if message is None:
            message = "The parameter values are inconsistent. Please refer to the docs."
        super().__init__(message, *args)


class ParamKinds(Enum):
    """
    An enumeration of the kinds of parameters supported by Flight.
    """

    NUMPY = auto()
    """
    Parameters implemented as NumPy `ndarray`s.
    """

    TORCH = auto()
    """
    Parameters implemented as PyTorch `Tensor`s.
    """


def infer_param_kind(param: t.Any) -> ParamKinds:
    """
    Detect the kind of parameter.

    Args:
        param (t.Any): The parameter to infer the type for.

    Returns:
        The kind of parameter.

    Throws:
        - `UnsupportedParameterKindError`: If the parameter kind is unknown/unsupported.
    """
    if isinstance(param, np.ndarray):
        return ParamKinds.NUMPY
    elif isinstance(param, torch.Tensor):
        return ParamKinds.TORCH
    else:
        raise UnsupportedParameterKindError()


def validate_param_kind(params: dict[str, t.Any]) -> ParamKinds:
    """
    Validate the kind of parameters.

    This function returns the kind of parameters (similar to `infer_param_kind`), but
    it will throw an error in the case where the parameters are not of the same kind.

    Args:
        params:

    Returns:
        The kind of parameters if they are of the same kind. Otherwise, an error is
        thrown.

    Throws:
        - `InconsistentParamValuesError`: If the parameter values are inconsistent.
        - `UnsupportedParameterKindError`: If the parameter kind is unknown/unsupported.
            This will be thrown by the `infer_param_kind` function.
    """
    param_kinds = set(map(infer_param_kind, params.values()))
    if len(param_kinds) != 1:
        raise InconsistentParamValuesError()
    return param_kinds.pop()


class Params(OrderedDict):
    """
    A wrapper class for model parameters, implemented as an `OrderedDict`.

    Throws:
        - `InconsistentParamValuesError`: If the parameter values are inconsistent.
        - `UnsupportedParameterKindError`: If the parameter kind is unknown/unsupported.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def numpy(self) -> NpParams:
        """
        Convert the parameters to NumPy `ndarray`s.

        Returns:
            The parameters in NumPy `ndarray`s.
        """
        match self.inferred_kind:
            case ParamKinds.NUMPY:
                return self
            case ParamKinds.TORCH:
                return OrderedDict((k, v.numpy()) for k, v in self.items())

    def torch(self) -> TorchParams:
        """
        Convert the parameters to PyTorch `Tensor`s.

        Returns:
            The parameters in the PyTorch `Tensor`s.
        """
        match self.inferred_kind:
            case ParamKinds.TORCH:
                return self
            case ParamKinds.NUMPY:
                return OrderedDict((k, torch.from_numpy(v)) for k, v in self.items())

    @functools.cached_property
    def inferred_kind(self) -> ParamKinds:
        """
        The inferred kind of the parameters.

        Returns:
            The kind of parameters.
        """
        return validate_param_kind(self)
