import pytest
import torch

from flight.learning.base import AbstractModule
from flight.learning.torch import TorchModule

SEED = 42


@pytest.fixture
def valid_module():
    class TestModule(TorchModule):
        def __init__(self):
            super().__init__()
            torch.manual_seed(SEED)
            self.m = torch.nn.Parameter(torch.tensor([1.0]))
            self.b = torch.nn.Parameter(torch.tensor([3.0]))

        def forward(self, x):
            return self.m * x + self.b

        def training_step(self, batch, batch_nb):
            return self(batch)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    return TestModule


@pytest.fixture
def invalid_module():
    class TestModule(TorchModule):  # noqa
        def __init__(self):
            super().__init__()
            torch.manual_seed(SEED)
            self.m = torch.nn.Parameter(torch.tensor([1.0]))
            self.b = torch.nn.Parameter(torch.tensor([3.0]))

        def forward(self, x):
            return self.m * x + self.b

    return TestModule


class TestModelInit:
    def test_valid_model_init(self, valid_module):
        model = valid_module()
        assert isinstance(model, TorchModule)
        assert isinstance(model, AbstractModule)
        assert isinstance(model, torch.nn.Module)

        x = torch.tensor([[1.0]])
        y = model(x)
        assert isinstance(y, torch.Tensor)

    def test_invalid_model_init(self, invalid_module):
        with pytest.raises(TypeError):
            invalid_module()

    def test_get_and_set_params(self, valid_module):
        model = valid_module()
        try:
            _ = model.get_params()
            params = model.get_params()
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
            model.set_params(params)
        except Exception as exc:
            print(exc)
            pytest.fail(exc, "Unexpected error/exception for `set_params()`.")

    def test_model_get_params(self, valid_module):
        model = valid_module()
        try:
            _ = model.get_params()
            params = model.get_params()
            print(params)
            assert "m" in params
            assert "b" in params
            model.set_params(params)
        except Exception as exc:
            print(exc)
            pytest.fail(exc, "Unexpected error/exception.")
