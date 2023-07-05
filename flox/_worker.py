import lightning as L
import torch

from typing import Any


class WorkerLogic:
    def __init__(self):
        pass

    def on_model_recv(self):
        pass

    def on_data_fetch(self, data, non_iid: bool = False):
        pass

    def on_model_fit(self, module: L.LightningModule, data_loader):
        trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=3
        )
        trainer.fit(module, data_loader)
        return module

    def on_model_send(self) -> dict[str, Any]:
        pass


class MnistWorkerLogic(WorkerLogic):
    def __init__(self, indices):
        super().__init__()
        self.name = "mnist"
        self.indices = []

    def on_data_fetch(self, data, non_iid: bool = False):
        from torch.utils.data import Subset
        from torchvision.datasets import MNIST
        from os import environ

        root = environ.get("PATH_DATASETS", ".")
        data = MNIST(root, download=True, train=True)
        data = Subset(data, indices=self.indices)
        return data
