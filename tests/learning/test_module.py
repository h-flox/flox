import collections as c

import numpy as np
import pickle
import pytest
import torch

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from flight.learning.module import *


@pytest.fixture
def param_data() -> list[float]:
    return [float(i) for i in range(10)]


def test_params_cls(param_data):
    weights = np.array(param_data)
    params = Params({f"p{i}": weights for i in range(10)})

    params_torch = params.torch()
    params_numpy = params.numpy()

    assert isinstance(params_numpy, dict)
    assert isinstance(params_torch, dict)

    assert isinstance(params_numpy, c.OrderedDict)
    assert isinstance(params_torch, c.OrderedDict)

    for key in params:
        assert isinstance(params_numpy[key], np.ndarray)
        assert isinstance(params_torch[key], torch.Tensor)
        assert float(params_numpy[key][0] == float(params_torch[key][0]))


def test_validate_numpy_params(param_data):
    weights = np.array(param_data)
    params = {f"p{i}": weights for i in range(10)}
    assert infer_param_kind(weights) == ParamKinds.NUMPY
    assert validate_param_kind(params) == ParamKinds.NUMPY


def test_validate_torch_params(param_data):
    weights = torch.tensor(param_data)
    params = {f"p{i}": weights for i in range(10)}
    assert infer_param_kind(weights) == ParamKinds.TORCH
    assert validate_param_kind(params) == ParamKinds.TORCH


################################################################################


class TestModule(TorchModule):
    def forward(self, x):
        return self.model(x)

    def configure_criterion(self):
        return torch.nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_valid_torch_module(param_data):
    # PART 1: Ensure initialization of `TorchModule` works.
    class MyModule(TestModule):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(1, 1)

    m = MyModule()
    for cls in [MyModule, TorchModule, torch.nn.Module]:
        assert isinstance(m, cls)

    # PART 2: Ensure parameters from `get_params` works.
    params = m.get_params()
    numpy_params = params.numpy()
    torch_params = params.torch()

    for key in params:
        assert isinstance(numpy_params[key], np.ndarray)
        assert isinstance(torch_params[key], torch.Tensor)

    result = m.set_params(numpy_params)
    assert result is None


def test_invalid_torch_module():
    class MyModule(TestModule):
        """
        This class should fail on initialization due
        to a missing call to `super().__init__()`.
        """

        def __init__(self):
            self.model = torch.nn.Linear(1, 1)

    with pytest.raises(AttributeError):
        MyModule()


################################################################################


class InMemoryDataModule(TorchDataModule):
    def __init__(self, data: TensorDataset):
        super().__init__()
        self.data = data

    def train_data(self) -> DataLoader:
        return DataLoader(self.data, shuffle=False)


class DiscDataModule(TorchDataModule):
    def __init__(self, root: Path | str | None):
        super().__init__()
        self.root = root

    def train_data(self) -> DataLoader:
        with open(self.root, "rb") as fp:
            data = pickle.load(fp)
        return DataLoader(data, shuffle=False)

@pytest.fixture
def data() -> TensorDataset:
    x = torch.tensor([[val] for val in range(100)])
    y = torch.tensor([[xi**2] for xi in x])
    return TensorDataset(x, y)

def test_in_memory_data(data):
    """
    Ensure that creating and initializing an in-memory `TorchDataModule` works.
    """
    dataset = InMemoryDataModule(data)
    assert isinstance(dataset, TorchDataModule)
    assert isinstance(dataset.train_data(), DataLoader)

def test_read_from_disc(data, tmp_path):
    """
    Ensure that creating and initializing a `TorchDataModule` from disc works. 
    """
    filename = tmp_path / "temp.pkl"
    with open(filename, "wb") as fp:
        pickle.dump(data, fp)

    dataset = DiscDataModule(filename)
    assert isinstance(dataset.train_data(), DataLoader)
    assert isinstance(dataset.train_data(), DataLoader)

    first_batch = next(iter(dataset.train_data()))
    assert first_batch[0].item() == 0
