import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from flight.federation.topologies.node import Node, WorkerState
from flight.learning.modules.torch import TorchDataModule
from flight.learning.torch import TorchModule
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
def module_cls() -> type[TorchModule]:
    class MyModule(TorchModule):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(1, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
            )

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_nb):
            inputs, targets = batch
            preds = self(inputs)
            loss = F.l1_loss(preds, targets)
            return loss

        def validation_step(self, batch, batch_nb):
            return self.training_step(batch, batch_nb)

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.SGD(self.parameters(), lr=0.001)

    return MyModule


@pytest.fixture
def data_cls() -> type[TorchDataModule]:
    class MyDataModule(TorchDataModule):
        def __init__(self):
            super().__init__()

            self.num_samples = 10_000
            self.num_features = 1
            seed = 0

            torch.manual_seed(seed)
            x = torch.randn((self.num_samples, self.num_features))
            y = torch.randn((self.num_samples, 1))
            self.raw_data = TensorDataset(x, y)

        def train_data(self, node: Node | None = None) -> DataLoader:
            subset = Subset(self.raw_data, indices=list(range(0, 8_000)))
            return DataLoader(subset, batch_size=32)

        def valid_data(self, node: Node | None = None) -> DataLoader | None:
            subset = Subset(self.raw_data, indices=list(range(8_000, 9_000)))
            return DataLoader(subset, batch_size=32)

        def test_data(self, node: Node | None = None) -> DataLoader | None:
            subset = Subset(self.raw_data, indices=list(range(9_000, 10_000)))
            return DataLoader(subset, batch_size=32)

    return MyDataModule


class TestTrainer:
    def test_torch_trainer_simple_init(self, node, worker_state, module_cls, data_cls):
        """
        Tests a basic setup of using the `TorchTrainer` class for PyTorch-based models.
        """
        trainer = TorchTrainer()
        assert isinstance(trainer, TorchTrainer)

        trainer = TorchTrainer(max_epochs=1)
        assert isinstance(trainer, TorchTrainer)

    def test_torch_trainer_node_init(self, node, worker_state, module_cls, data_cls):
        """
        Tests a basic setup of using the `TorchTrainer` class for PyTorch-based models.
        """
        model = module_cls()
        data = data_cls()
        trainer = TorchTrainer(
            node=node, strategy=DefaultTrainerStrategy(), max_epochs=1
        )
        assert isinstance(model, TorchModule)
        assert isinstance(trainer, TorchTrainer)

        results = trainer.fit(worker_state, model, data)
        assert isinstance(results, list)

    def test_node_device_specifier(self, node):
        """Confirms that the device"""
        trainer = TorchTrainer(
            node=node, strategy=DefaultTrainerStrategy(), max_epochs=1
        )
        assert str(trainer._device) == "cpu"

        node["device"] = "cuda"
        trainer = TorchTrainer(
            node=node, strategy=DefaultTrainerStrategy(), max_epochs=1
        )
        assert str(trainer._device) == "cuda"

        node["device"] = "mps"
        trainer = TorchTrainer(
            node=node, strategy=DefaultTrainerStrategy(), max_epochs=1
        )
        assert str(trainer._device) == "mps"

    def test_data_module(self, node):
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

            def train_data(self, _: Node | None = None):
                return DataLoader(self.train, batch_size=8)

        assert isinstance(Foo().train_data(node), DataLoader)
