import torch

from torch.utils.data import DataLoader

from flox.worker.model import FloxModel
from flox.worker.update import LocalUpdate


class LocalTrainer:
    def fit(
        self, model: FloxModel, train_dataloader: DataLoader, val_dataloader: DataLoader
    ) -> LocalUpdate:
        model.train()
        optimizer = model.configure_optimizers()
        torch.set_grad_enabled(True)

        losses = []
        for idx, batch in enumerate(train_dataloader):
            loss = model.training_step(batch, idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return LocalUpdate()
