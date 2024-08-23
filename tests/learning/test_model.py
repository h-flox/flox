import pytest
import torch

from flight.learning.modules.prototypes import HasParameters
from flight.learning.torch import TorchModule

SEED = 42


@pytest.fixture
def valid_module():
    class TestModule(TorchModule):
        def __init__(self):
            super().__init__()
            torch.manual_seed(SEED)
            self.m = torch.nn.Parameter(torch.randn(1))
            self.b = torch.nn.Parameter(torch.randn(1))

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
            self.m = torch.nn.Parameter(torch.randn(1))
            self.b = torch.nn.Parameter(torch.randn(1))

        def forward(self, x):
            return self.m * x + self.b

    return TestModule


class TestModelInit:
    def test_valid_model_init(self, valid_module):
        model = valid_module()
        assert isinstance(model, TorchModule)
        assert isinstance(model, HasParameters)
        assert isinstance(model, torch.nn.Module)

        x = torch.tensor([[1.0]])
        y = model(x)
        assert isinstance(y, torch.Tensor)

    def test_invalid_model_init(self, invalid_module):
        with pytest.raises(TypeError):
            invalid_module()

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
