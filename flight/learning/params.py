import typing as t
from enum import Enum, auto

import numpy as np
import torch

from flight.learning import NpParams, TorchParams


class UnsupportedParameterKindError(ValueError):
    """An Exception raised when an unsupported parameter kind is detected."""

    def __init__(self, message: str | None = None, *args):
        if message is None:
            message = (
                "The parameter kind is unknown or unsupported. "
                "Please refer to the docs."
            )
        super().__init__(message, *args)


class InconsistentParamValuesError(ValueError):
    """An Exception raised when the parameter value kinds are inconsistent."""

    def __init__(self, message: str | None = None, *args):
        if message is None:
            message = "The parameter values are inconsistent. Please refer to the docs."
        super().__init__(message, *args)


class ParamKinds(Enum):
    NUMPY = auto()
    TORCH = auto()


def infer_param_kind(param: t.Any) -> ParamKinds:
    """
    Detect the kind of parameter.

    Args:
        param (t.Any): The parameter to infer the type for.

    Returns:
        The kind of parameter.

    Throws:
        - `ValueError`: If the parameter kind is unknown or unsupported.
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
    """
    param_kinds = set(map(infer_param_kind, params.values()))
    if len(param_kinds) != 1:
        raise InconsistentParamValuesError()
    return param_kinds.pop()


class Params:
    def __init__(self, raw_params: dict[str, t.Any]):
        self._raw_params = raw_params
        self._inferred_kind = validate_param_kind(raw_params)

    def numpy(self) -> NpParams:
        match self._inferred_kind:
            case ParamKinds.NUMPY:
                return self._raw_params
            case ParamKinds.TORCH:
                return {k: v.numpy() for k, v in self._raw_params.items()}

    def torch(self) -> TorchParams:
        match self._inferred_kind:
            case ParamKinds.TORCH:
                return self._raw_params
            case ParamKinds.NUMPY:
                return {k: torch.from_numpy(v) for k, v in self._raw_params.items()}


# class NpParams(Params):
#     @abc.abstractmethod
#     def numpy(self) -> dict[str, npt.NDArray]:
#         pass
#
#
# class TorchParams(Params):
#     @abc.abstractmethod
#     def numpy(self) -> dict[str, npt.NDArray]:
#         pass
