from __future__ import annotations

import abc
import collections as c
import enum
import typing as t

import numpy as np
import torch

_T = t.TypeVar("_T", np.ndarray, torch.Tensor)
_DEFAULT_INCLUDE_STATE: t.Final[bool] = False


class ParamKinds(str, enum.Enum):
    """
    An enumeration of the kinds of parameters supported by Flight.
    """

    NUMPY = "numpy"
    """
    Parameters implemented as NumPy `ndarray`s.
    """

    TORCH = "torch"
    """
    Parameters implemented as PyTorch `Tensor`s.
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


def verify_consistent_value_types(d: t.Mapping) -> bool:
    """
    Verify that all values in the mapping are of the same type.

    Args:
        d (Mapping[Any, Any]): The mapping to check.

    Returns:
        `True` if all values are of the same type, `False` otherwise.
    """
    if not d:
        return True
    first_value_type = type(next(iter(d.values())))
    return all(isinstance(value, first_value_type) for value in d.values())


def verify_correct_value_types(d: t.Mapping, value_type: t.Type[t.Any]) -> bool:
    """
    Verify that all values in the mapping are of the specified type.

    Args:
        d (Mapping[Any, Any]): The mapping to check.
        value_type (type): The type to check against.

    Returns:
        bool: True if all values are of the specified type, False otherwise.
    """
    return all(isinstance(value, value_type) for value in d.values())


#######################################################################################


class Params(abc.ABC, c.OrderedDict[str, _T]):
    def __new__(cls, *args, **kwargs):
        if hasattr(cls, "__abstractmethods__") and len(cls.__abstractmethods__) > 0:
            raise TypeError(
                f"Can't instantiate abstract class {cls.__name__} with "
                f"abstract methods {', '.join(cls.__abstractmethods__)}"
            )

        return super().__new__(cls)

    @abc.abstractmethod
    def numpy(self) -> NumpyParams:
        """
        Convert the parameters to a `NumPyParams` instance.

        Returns:
            An instance of `NumpyParams` containing the parameters.
        """

    @abc.abstractmethod
    def torch(self) -> TorchParams:
        """
        Convert the parameters to a `TorchParams` instance.

        Returns:
            An instance of `TorchParams` containing the parameters.
        """

    @property
    @abc.abstractmethod
    def kind(self) -> ParamKinds:
        """
        The kind of parameters represented by this instance.

        Returns:
            An instance of `ParamKinds` representing the kind of parameters.
        """


ParamLike = t.Union[np.ndarray, torch.Tensor, list[float | int]]

OrderedDictLikeParams: t.TypeAlias = t.Union[
    t.Iterable[tuple[str, ParamLike]],
    t.Mapping[str, ParamLike],
]


def parameters(
    args: OrderedDictLikeParams,
    backend: str = "numpy",
) -> Params:
    """
    Create a new `Params` instance from the given data.

    Specifically, this method does some light parsing to ensure that the given
    data is in a format that can supports mathematical operations. Data should be

    This is the preferred way to create a `Params` instance, if you are doing it
    manually (i.e., not from calling [`get_params()`]
    [flight.learning.module.TorchModule.get_params]).

    Args:
        args (OrderedDictLikeParams):
            The data to create the parameters from. Each tuple should contain a key
            (str) and a value (list of floats, ints, or a numpy array).
        backend (str): The backend to use for the parameters. Supported backends are
            "numpy" and "torch". Defaults to "numpy".

    Returns:
        An instance of `Params` containing the parameters.

        This depends on the `backend` argument: by default, it returns a
        `NumpyParams` instance, but if `backend` is set to "torch", it returns
        a `TorchParams` instance.

    Examples:
        >>> dat = {"a": [1, 2, 3, 4]}
        >>> parameters(dat, backend="auto")
        NumpyParams([('a', array([1., 2., 3., 4.]))])
        >>> parameters(dat, backend="torch")
        TorchParams([('a', tensor([1., 2., 3., 4.], dtype=torch.float64))])
    """
    data = c.OrderedDict(args)
    inferred_kind = type(next(iter(data.values())))

    if inferred_kind is list:
        # We do this to get the function to fall back to the `NumpyParams` class.
        # For some reason, when trying to add cases for the `list` type, the match
        # statement would not match the `list` type correctly.
        inferred_kind = np.ndarray
        data = c.OrderedDict(
            {key: np.array(value, dtype=float) for key, value in data.items()}
        )

    if not verify_consistent_value_types(data):
        raise InconsistentParamValuesError(
            "All values in the parameters must be of the same type."
        )

    match inferred_kind, backend:
        case np.ndarray, "auto":
            return NumpyParams(data)

        case np.ndarray, "numpy":
            return NumpyParams(data)

        case np.ndarray, "torch":
            return NumpyParams(data).torch()

        case torch.Tensor, "auto":
            return TorchParams(data)

        case torch.Tensor, "numpy":
            return TorchParams(data).numpy()

        case torch.Tensor, "torch":
            return TorchParams(data)

        case _:
            raise UnsupportedParameterKindError(
                f"Unsupported backend '{backend}'. Supported backends are 'auto', "
                f"'numpy', and 'torch'. The inferred kind was `{inferred_kind}`."
            )


class NumpyParams(Params):
    """
    A class representing parameters as NumPy arrays.
    """

    def __init__(self, *args, skip_validation: bool = False):
        super().__init__(*args)
        if skip_validation:
            return

        if not verify_correct_value_types(self, np.ndarray):
            raise InconsistentParamValuesError(
                "All values in NumpyParams must be numpy.ndarray."
            )

    def numpy(self) -> NumpyParams:
        return self

    def torch(self) -> TorchParams:
        return TorchParams((k, torch.from_numpy(v)) for k, v in self.items())

    @property
    def kind(self):
        return ParamKinds.NUMPY


class TorchParams(Params):
    """
    A class representing parameters as PyTorch tensors.
    """

    def __init__(self, *args, skip_validation: bool = False):
        super().__init__(*args)
        if skip_validation:
            return

        if not verify_correct_value_types(self, torch.Tensor):
            raise InconsistentParamValuesError(
                "All values in NumpyParams must be numpy.ndarray."
            )

    def numpy(self) -> NumpyParams:
        return NumpyParams((k, v.detach().cpu().numpy()) for k, v in self.items())

    def torch(self) -> TorchParams:
        return self

    @property
    def kind(self):
        return ParamKinds.TORCH
