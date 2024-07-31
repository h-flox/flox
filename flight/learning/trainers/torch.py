from __future__ import annotations

import typing as t

import torch
from torch.utils.data import DataLoader

from ..modules.torch import FlightModule, TorchDataModule

if t.TYPE_CHECKING:
    from ...federation.topologies.node import Node, WorkerState
    from ...strategies.trainer import TrainerStrategy

    EVAL_DATALOADERS: t.TypeAlias = t.Any  #
    TRAIN_DATALOADERS: t.TypeAlias = t.Any  #
    _PATH: t.TypeAlias = t.Any  #
    LightningDataModule: t.TypeAlias = t.Any  #


class TorchTrainer:
    def __init__(self, node: Node, strategy: TrainerStrategy, max_epochs: int):
        self.node = node
        self.strategy = strategy
        self.max_epochs = max_epochs
        self._device = torch.device(node.extra.get("device", "cpu"))
        # self.logger =

    def fit(
        self,
        node_state: WorkerState,
        model: FlightModule,
        data: TorchDataModule,
        validate_every_n_epochs: int = 1,
        # train_dataloaders: TRAIN_DATALOADERS | LightningDataModule | None = None,
        # val_dataloaders: EVAL_DATALOADERS | None = None,
        # datamodule: LightningDataModule | None = None,
        ckpt_path: _PATH | None = None,
    ):
        """

        Args:
            node_state:
            model:
            data:
            validate_every_n_epochs:
            ckpt_path:

        Raises:
            - ValueError: Thrown when illegal values are given to arguments.

        Returns:

        """
        # TODO: Run the argument validation in a separate utility function to keep this function light.
        if validate_every_n_epochs < 1:
            raise ValueError("Illegal value for argument `validate_every_n_epochs`.")

        model.to(self._device)

        results = []

        train_dataloader = data.train_data(self.node)
        valid_dataloader = data.valid_data(self.node)

        if not isinstance(train_dataloader, DataLoader):
            raise TypeError(
                "Method for argument `data.train_data(.)` must return a `DataLoader`."
            )
        if not isinstance(valid_dataloader, DataLoader):
            raise TypeError(
                "Method for argument `data.valid_data(.)` must return a `DataLoader`."
            )

        optimizer = model.configure_optimizers()

        for epoch in range(self.max_epochs):
            print(f"❯ Running epoch {epoch} out of {self.max_epochs}.")
            train_losses = self._epoch(
                node_state,
                model,
                optimizer,
                train_dataloader,
                # train_dataloaders,
            )
            for l in train_losses:
                results.append({"epoch": epoch, "train/loss": l.item()})

            to_validate = all(
                [epoch % validate_every_n_epochs == 0, valid_dataloader is not None]
            )
            if to_validate:
                val_losses = self.validate(model, valid_dataloader)
                for l in val_losses:
                    results.append({"epoch": epoch, "val/loss": l.item()})

        return results

    def _epoch(
        self,
        node_state: WorkerState,
        model: FlightModule,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
    ):
        self._set_train_mode(model, True)

        losses = []
        for batch_idx, batch in enumerate(dataloader):
            batch = self._batch_to_device(batch)
            loss = model.training_step(batch, batch_idx)

            # Perform backpropagation and call trainer strategy callbacks.
            optimizer.zero_grad()
            loss = self.strategy.before_backprop(node_state, loss)
            loss.backward()
            loss = self.strategy.after_backprop(node_state, loss)
            losses.append(loss)
            optimizer.step()

        return losses

    def validate(self, model: FlightModule, dataloader: DataLoader, *args, **kwargs):
        self._set_train_mode(model, False)

        losses = []
        for batch_idx, batch in enumerate(dataloader):
            batch = self._batch_to_device(batch)
            loss = model.validation_step(batch, batch_idx)
            losses.append(loss)

        return losses

    def _batch_to_device(self, batch: tuple[t.Any, ...]):
        items = []
        for item in batch:
            try:
                item = item.to(self._device)
            except AttributeError:
                pass
            items.append(item)
        return tuple(items)
        # return tuple(item.to(self._device) for item in batch)

    def _set_train_mode(self, model, mode: True):
        torch.set_grad_enabled(mode)
        if mode:
            model.train()
        else:
            model.eval()
