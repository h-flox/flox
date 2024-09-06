import pytest
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule, LightningDataModule
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from flight.federation.topologies.node import Node, NodeKind
from flight.learning.lightning import LightningTrainer


@pytest.fixture
def node() -> Node:
    node = Node(idx=0, kind=NodeKind.WORKER)
    return node


@pytest.fixture
def module_light_cls() -> type[LightningModule]:
    class SimpleLightningModule(LightningModule):
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
            self.logger.log_metrics({"loss": loss.item()})
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

    return SimpleLightningModule


@pytest.fixture
def data_cls() -> type[LightningDataModule]:
    class SimpleLightningDataModule(LightningDataModule):
        def __init__(self):
            super().__init__()

            self.num_samples = 10_000
            self.num_features = 1

            g = torch.Generator(device="cpu")
            g.manual_seed(0)

            x = torch.randn((self.num_samples, self.num_features), generator=g)
            y = torch.randn((self.num_samples, 1), generator=g)
            self.raw_data = TensorDataset(x, y)

        def train_dataloader(self) -> DataLoader:
            subset = Subset(self.raw_data, indices=list(range(0, 8_000)))
            return DataLoader(subset, batch_size=32)

        def val_dataloader(self) -> DataLoader | None:
            subset = Subset(self.raw_data, indices=list(range(8_000, 9_000)))
            return DataLoader(subset, batch_size=32)

        def test_dataloader(self) -> DataLoader | None:
            subset = Subset(self.raw_data, indices=list(range(9_000, 10_000)))
            return DataLoader(subset, batch_size=32)

    return SimpleLightningDataModule


class TestLightningTrainer:
    def test_lightning_trainer(self, node, module_light_cls, data_cls):
        """
        Tests a basic setup of using the `LightningTrainer` class for
        PyTorch-based models.
        """
        model = module_light_cls()
        data = data_cls()
        kwargs = {"max_epochs": 1, "log_every_n_steps": 100}
        trainer = LightningTrainer(node, **kwargs)

        assert isinstance(model, LightningModule)
        assert isinstance(trainer, LightningTrainer)
        assert isinstance(data, LightningDataModule)

        results = trainer.fit(model, data)

        assert isinstance(results, dict)

    def test_test_process(self, node, module_light_cls, data_cls):
        """
        Tests that no errors occur during basic testing.
        """
        model = module_light_cls()
        data = data_cls()
        kwargs = {"max_epochs": 1, "log_every_n_steps": 100}
        trainer = LightningTrainer(node, **kwargs)

        trainer.fit(model, data)
        records = trainer.test(model, data)

        assert isinstance(records, list)

    def test_val_process(self, node, module_light_cls, data_cls):
        """
        Tests that no errors occur during basic validation.
        """
        model = module_light_cls()
        data = data_cls()
        kwargs = {"max_epochs": 1, "log_every_n_steps": 100}
        trainer = LightningTrainer(node, **kwargs)

        trainer.fit(model, data)
        records = trainer.validate(model, data)

        assert isinstance(records, list)
