import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from flight.federation.topologies.node import Node, WorkerState
from flight.learning.datasets import DataLoadable
from flight.learning.modules.torch import FlightModule, TorchDataModule
from flight.learning.trainers.torch import TorchTrainer
from flight.strategies.base import DefaultTrainerStrategy

NUM_FEATURES = 10


@pytest.fixture
def node() -> Node:
    node = Node(idx=0, kind="worker")
    return node


@pytest.fixture
def worker_state() -> WorkerState:
    return WorkerState(0, None, None)


@pytest.fixture
def module_cls() -> type[FlightModule]:
    class MyModule(FlightModule):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(NUM_FEATURES, 100),
                torch.nn.Linear(100, 1),
            )

        def forward(self, x):
            return self.model(x)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-3)

        def training_step(self, batch, batch_nb):
            inputs, targets = batch
            preds = self(inputs)
            return F.l1_loss(preds, targets)

    return MyModule


@pytest.fixture
def data_cls() -> type[DataLoadable]:
    class MyDataLoadable(DataLoadable):
        def __init__(self):
            torch.manual_seed(0)
            n = 1000  # total number of data samples
            f = NUM_FEATURES  # number of features
            n_train, n_test, n_valid = n * 0.8, n * 0.1, n * 0.1
            n_train, n_test, n_valid = int(n_train), int(n_test), int(n_valid)
            self.train = TensorDataset(
                torch.randn((n_train, f)),
                torch.randn((n_train, 1)),
            )
            self.test = TensorDataset(
                torch.randn((n_test, f)),
                torch.randn((n_test, 1)),
            )
            self.valid = TensorDataset(
                torch.randn((n_valid, f)),
                torch.randn((n_valid, 1)),
            )

        def load(self, node: Node, mode: str):
            match mode:
                case "train":
                    return self.train
                case "test":
                    return self.test
                case "valid" | "validation":
                    return self.valid
                case _:
                    raise ValueError("Illegal `mode` literal value.")

    return MyDataLoadable


class TestTrainer:
    def _default_torch_trainer(self, node, worker_state, module_cls, data_cls):
        """Tests a basic setup of using the `TorchTrainer` class for PyTorch-based models."""
        model = module_cls()
        data = data_cls()
        trainer = TorchTrainer(node, DefaultTrainerStrategy(), 1)
        assert isinstance(model, FlightModule)
        assert isinstance(trainer, TorchTrainer)

        trainer.fit(worker_state, model, data)

    def test_node_device_specifier(self, node):
        """Confirms that the device"""
        trainer = TorchTrainer(node, DefaultTrainerStrategy(), 1)
        assert str(trainer._device) == "cpu"

        node.extra["device"] = "cuda"
        trainer = TorchTrainer(node, DefaultTrainerStrategy(), 1)
        assert str(trainer._device) == "cuda"

        node.extra["device"] = "mps"
        trainer = TorchTrainer(node, DefaultTrainerStrategy(), 1)
        assert str(trainer._device) == "mps"

    def test_temp(self, node):
        class Foo(TorchDataModule):
            def __init__(self):
                super().__init__()
                torch.manual_seed(0)
                n = 1000  # total number of data samples
                f = NUM_FEATURES  # number of features
                n_train, n_test, n_valid = n * 0.8, n * 0.1, n * 0.1
                n_train, n_test, n_valid = int(n_train), int(n_test), int(n_valid)
                self.train = TensorDataset(
                    torch.randn((n_train, f)),
                    torch.randn((n_train, 1)),
                )
                self.test = TensorDataset(
                    torch.randn((n_test, f)),
                    torch.randn((n_test, 1)),
                )
                self.valid = TensorDataset(
                    torch.randn((n_valid, f)),
                    torch.randn((n_valid, 1)),
                )

            def train_data(self, node: Node | None = None):
                return DataLoader(self.train, batch_size=8)

        assert isinstance(Foo().train_data(node), DataLoader)
