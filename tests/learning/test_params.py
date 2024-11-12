import numpy as np
import pytest
import torch

from flight.learning.params import (
    validate_param_kind,
    ParamKinds,
    infer_param_kind,
    InconsistentParamValuesError,
)


@pytest.fixture
def param_data() -> list[float]:
    return [0.0, 1.0, 2.0]


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
