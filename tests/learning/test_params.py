from collections import OrderedDict

import numpy as np
import pytest
import torch

from flight.learning.params import (
    validate_param_kind,
    Params,
    ParamKinds,
    infer_param_kind,
    InconsistentParamValuesError,
)


@pytest.fixture
def param_data() -> list[float]:
    return [0.0, 1.0, 2.0]


def test_params_cls(param_data):
    p = np.array(param_data)
    params = Params({f"p{i}": p for i in range(10)})

    params_torch = params.torch()
    params_np = params.numpy()

    assert isinstance(params_np, dict)
    assert isinstance(params_torch, dict)

    assert isinstance(params_np, OrderedDict)
    assert isinstance(params_torch, OrderedDict)

    key = next(iter(params))
    assert isinstance(params_np[key], np.ndarray)
    assert isinstance(params_torch[key], torch.Tensor)
    assert float(params_np[key][0]) == float(params_torch[key][0])


def test_validate_numpy_params(param_data):
    p = np.array(param_data)
    params = {f"p{i}": p for i in range(10)}
    assert infer_param_kind(p) == ParamKinds.NUMPY
    assert validate_param_kind(params) == ParamKinds.NUMPY


def test_validate_torch_params(param_data):
    p = torch.tensor(param_data)
    params = {f"p{i}": p for i in range(10)}
    assert infer_param_kind(p) == ParamKinds.TORCH
    assert validate_param_kind(params) == ParamKinds.TORCH


def test_inconsistent_params(param_data):
    with pytest.raises(InconsistentParamValuesError):
        bad_params = {"p0": torch.tensor(param_data), "p1": np.array(param_data)}
        validate_param_kind(bad_params)
