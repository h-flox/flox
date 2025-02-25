import torch
import pytest

from flight.learning.module import TorchModule


class InvalidModule(TorchModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.model(x)


def my_criterion(self):
    return torch.nn.MSELoss()


def my_optimizer(self):
    return torch.optim.SGD(self.parameters(), lr=0.01)


def test_invalid_torch_module():
    with pytest.raises(TypeError):
        module = InvalidModule()


def test_invalid_torch_module_with_criterion():
    class InvalidModuleWithCriterion(InvalidModule):
        configure_criterion = my_criterion

    with pytest.raises(TypeError):
        module = InvalidModuleWithCriterion()


def test_invalid_torch_module_with_optimizer():
    class InvalidModuleWithOptimizer(InvalidModule):
        configure_optimizers = my_optimizer

    with pytest.raises(TypeError):
        module = InvalidModuleWithOptimizer()


def test_valid_torch_module():

    class ValidModule(InvalidModule):
        configure_criterion = my_criterion
        configure_optimizers = my_optimizer

    assert isinstance(ValidModule(), TorchModule)
