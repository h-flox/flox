import pytest
from pathlib import Path

from lightning.pytorch import LightningModule, LightningDataModule
from lightning.pytorch.loggers import CSVLogger

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from flight.strategies.base import DefaultTrainerStrategy
from flight.learning.trainers.lightning import LightningTrainer
from flight.learning.modules.torch import FlightModule
from flight.federation.topologies.node import Node, WorkerState, NodeKind
from flight.learning.modules.prototypes import DataLoadable


@pytest.fixture
def node() -> Node:
    node = Node(idx=0, kind=NodeKind.WORKER)
    return node

@pytest.fixture
def module_light_cls() -> type[LightningModule]:
    class MyLightningModule(LightningModule):
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
        
        def test_step(self, batch, batch_nb):
            inputs, labels = batch
            prediction = self(inputs)
            loss = F.l1_loss(prediction, labels)

            _, prediction = torch.max(prediction, 1)

            return loss

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.SGD(self.parameters(), lr=0.001)

    return MyLightningModule

@pytest.fixture
def data_light_cls() -> type[DataLoadable]:
    class MyLightningDataModule(LightningDataModule):
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

    return MyLightningDataModule

@pytest.fixture
def log_cls() -> CSVLogger:
    logger = CSVLogger('logs')
    return logger

class TestLightningTrainer:
    def test_lightning_trainer(self, node, module_light_cls, data_light_cls):
        """
        Tests a basic setup of using the `LightningTrainer` class for PyTorch-based models.
        """
        model = module_light_cls()
        data = data_light_cls()
        trainer = LightningTrainer(node)

        assert isinstance(model, LightningModule)
        assert isinstance(trainer, LightningTrainer)
        assert isinstance(data, LightningDataModule)

        results = trainer.fit(model, data)
        assert isinstance(results, dict)

    def test_node_device_specifier(self, node):
        """Confirms that the device"""
        trainer = LightningTrainer(node)
        assert str(trainer._device) == "cpu"

        node["device"] = "cuda"
        trainer = LightningTrainer(node)
        assert str(trainer._device) == "cuda"

        node["device"] = "mps"
        trainer = LightningTrainer(node)
        assert str(trainer._device) == "mps"
    
    def test_test_process(self, node, module_light_cls, data_light_cls):
        model = module_light_cls()
        data = data_light_cls()
        trainer = LightningTrainer(node)

        trainer.fit(model, data)

        records = trainer.test(model, data)
        assert isinstance(records, list)
    
    def test_val_process(self, node, module_light_cls, data_light_cls):
        model = module_light_cls()
        data = data_light_cls()
        trainer = LightningTrainer(node)

        trainer.fit(model, data)
        records = trainer.validate(model, data)

        assert isinstance(records, list)

    



