import pytest
import torch

from flight.learning.base import AbstractModule
from flight.learning.torch import TorchModule
from testing.fixtures import invalid_module_cls, valid_module


def test_valid_model_init(valid_module):
    assert isinstance(valid_module, TorchModule)
    assert isinstance(valid_module, AbstractModule)
    assert isinstance(valid_module, torch.nn.Module)

    x = torch.tensor([[1.0]])
    y = valid_module(x)
    assert isinstance(y, torch.Tensor)


def test_invalid_model_init(invalid_module_cls):
    with pytest.raises(TypeError):
        invalid_module_cls()


def test_get_and_set_params(valid_module):
    try:
        _ = valid_module.get_params()
        params = valid_module.get_params()
        print(params)
        assert "m" in params
        assert "b" in params
        assert params["m"].item() == 1.0
        assert params["b"].item() == 3.0
    except Exception as exc:
        print(exc)
        pytest.fail(exc, "Unexpected error/exception for `get_params()`.")

    try:
        params["m"] = torch.tensor([10.0])
        params["b"] = torch.tensor([30.0])
        valid_module.set_params(params)
    except Exception as exc:
        print(exc)
        pytest.fail(exc, "Unexpected error/exception for `set_params()`.")


def test_model_get_params(valid_module):
    try:
        _ = valid_module.get_params()
        params = valid_module.get_params()
        print(params)
        assert "m" in params
        assert "b" in params
        valid_module.set_params(params)
    except Exception as exc:
        print(exc)
        pytest.fail(exc, "Unexpected error/exception.")
