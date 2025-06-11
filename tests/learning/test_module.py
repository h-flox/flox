import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from flight.learning.module import (
    TorchDataModule,
    TorchModule,
)


################################################################################


class SimpleTestModule(TorchModule):
    def forward(self, x):
        return self.model(x)

    def configure_criterion(self):
        return torch.nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_valid_torch_module():
    # PART 1: Ensure initialization of `TorchModule` works.
    class MyModule(SimpleTestModule):
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

    result = m.set_params(numpy_params)  # noqa
    assert result is None


def test_invalid_torch_module():
    class MyModule(SimpleTestModule):
        """
        This class should fail on initialization due
        to a missing call to `super().__init__()`.
        """

        def __init__(self):  # noqa
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
