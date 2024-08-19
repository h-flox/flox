import pytest
import os

from lightning.pytorch import LightningModule, LightningDataModule

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.optim import SGD
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from flight.learning.trainers.lightning import LightningTrainer
from flight.federation.topologies.node import Node, NodeKind


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
            self.logger.log_metrics({'loss': loss})
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
def data_light_cls() -> type[LightningDataModule]:
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

        def train_dataloader(self) -> DataLoader:
            subset = Subset(self.raw_data, indices=list(range(0, 8_000)))
            return DataLoader(subset, batch_size=32)

        def valid_dataloader(self) -> DataLoader | None:
            subset = Subset(self.raw_data, indices=list(range(8_000, 9_000)))
            return DataLoader(subset, batch_size=32)

        def test_dataloader(self) -> DataLoader | None:
            subset = Subset(self.raw_data, indices=list(range(9_000, 10_000)))
            return DataLoader(subset, batch_size=32)

    return MyLightningDataModule

@pytest.fixture
def data_cifar_cls() -> type[LightningDataModule]:
    class CifarDataModule(LightningDataModule):
        def __init__(self) -> None:
            train_data = CIFAR10(root=os.environ["CIFAR_DATASET"], train=True, transform=transforms.ToTensor())
            self.train_subset = Subset(train_data, list(range(100)))

            test_data = CIFAR10(root=os.environ["CIFAR_DATASET"], train=False, transform=transforms.ToTensor())
            self.test_subset = Subset(test_data, list(range(100)))
            self.val_subset = Subset(test_data, list(range(100,200)))
        
        def train_dataloader(self):
            return DataLoader(self.train_subset, shuffle=True)

        def test_dataloader(self):
            return DataLoader(self.test_subset, shuffle=True)
        
        def valid_dataloader(self):
            return DataLoader(self.val_subset, shuffle=True)
        
    return CifarDataModule

@pytest.fixture
def module_cifar_lightning() -> type[LightningModule]:
    class MyCifarModule(LightningModule):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3,6,5)
            self.pool = nn.MaxPool2d(2,2)
            self.conv2 = nn.Conv2d(6,16,5)
            self.fc1 = nn.Linear(16*5*5,120)
            self.fc2 = nn.Linear(120,84)
            self.fc3 = nn.Linear(84,10)
            self.hidden_activation = nn.ReLU()

            self.records = []
            self.train_count = 0
            self.val_count = 0
            self.test_count = 0

            self.running_loss = 0.0
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.hidden_activation(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.hidden_activation(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.hidden_activation(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

        
        def training_step(self, batch, batch_idx):
            if batch_idx == 0:
                self.running_loss = 0.0
                
            self.train_count += 1
            inputs, labels = batch
            prediction = self(inputs)
            loss = F.cross_entropy(prediction, labels)

            self.running_loss += loss.item()
            if batch_idx % 20 == 0:
                self.log(name='train/loss', value=self.running_loss/20)
                #self.logger.log_metrics({'train/loss': self.running_loss/20.0})
                self.running_loss = 0.0
            return loss

        def test_step(self, batch, batch_idx):
            self.test_count += 1
            inputs, labels = batch
            prediction = self(inputs)
            loss = F.cross_entropy(prediction, labels)

            _, prediction = torch.max(prediction, 1)
            test_acc = (labels == prediction).float().mean().item()

            self.log(name='test/acc', value=test_acc)
            return loss

        def validation_step(self, batch, batch_idx):
            self.val_count += 1
            inputs, labels = batch
            prediction = self(inputs)
            loss = F.cross_entropy(prediction, labels)

            _, prediction = torch.max(prediction, 1)
            val_acc = (labels == prediction).float().mean().item()

            if batch_idx % 20 == 0:
                self.log(name='val/loss', value=loss)
                #self.logger.log_metrics({'val/loss': loss}) 
            return loss

        def configure_optimizers(self):
            optimizer = SGD(self.parameters(), lr=0.01)
            return optimizer
    return MyCifarModule

class TestLightningTrainer:
    def test_lightning_trainer(self, node, module_light_cls, data_light_cls):
        """
        Tests a basic setup of using the `LightningTrainer` class for PyTorch-based models.
        """
        model = module_light_cls()
        data = data_light_cls()
        kwargs = {"max_epochs": 1, "log_every_n_steps": 100}
        trainer = LightningTrainer(node, **kwargs)

        assert isinstance(model, LightningModule)
        assert isinstance(trainer, LightningTrainer)
        assert isinstance(data, LightningDataModule)

        results = trainer.fit(model, data)
        
        assert isinstance(results, dict)

    def test_test_process(self, node, module_light_cls, data_light_cls):
        """
        Tests that no errors occur during basic testing.
        """
        model = module_light_cls()
        data = data_light_cls()
        kwargs = {"max_epochs": 1, "log_every_n_steps": 100}
        trainer = LightningTrainer(node, **kwargs)

        trainer.fit(model, data)

        records = trainer.test(model, data)
        assert isinstance(records, list)

    def test_val_process(self, node, module_light_cls, data_light_cls):
        """
        Tests that no errors occur during basic validation.
        """
        model = module_light_cls()
        data = data_light_cls()
        kwargs = {"max_epochs": 1, "log_every_n_steps": 100}
        trainer = LightningTrainer(node, **kwargs)

        trainer.fit(model, data)
        records = trainer.validate(model, data)

        assert isinstance(records, list)
    
    def test_fit_cifar(self, node, module_cifar_lightning, data_cifar_cls):
        """
        Tests that the 'LightningTrainer' can train a classifier on the CIFAR10 dataset.
        """
        node = node
        model = module_cifar_lightning()
        data = data_cifar_cls()

        kwargs = {"max_epochs": 5, "log_every_n_steps": 10}
        trainer = LightningTrainer(node, **kwargs)

        train_record = trainer.fit(model, data)
        assert isinstance(train_record['train/loss'], torch.Tensor)

        test_record = trainer.test(model, data)

        print(test_record)