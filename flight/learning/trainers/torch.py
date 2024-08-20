from __future__ import annotations

import typing as t

import torch
import tqdm
from torch.utils.data import DataLoader

from ...types import Record
from ..modules.torch import TorchDataModule, TorchModule

if t.TYPE_CHECKING:
    from ...federation.topologies.node import Node, WorkerState
    from ...strategies.trainer import TrainerStrategy

    EVAL_DATALOADERS: t.TypeAlias = t.Any  #
    TRAIN_DATALOADERS: t.TypeAlias = t.Any  #
    _PATH: t.TypeAlias = t.Any  #
    LightningDataModule: t.TypeAlias = t.Any  #


class TorchTrainer:
    def __init__(
        self,
        max_epochs: int = 1,
        node: Node | None = None,
        strategy: TrainerStrategy | None = None,
        progress_bar: bool = True,
    ):
        self.node = node
        self.strategy = strategy
        self.max_epochs = max_epochs
        self._progress_bar = progress_bar
        self._curr_step = 0
        self._results = []

        if self.node is None:
            from flight.federation.topologies import Node

            self.node = Node(idx=0, kind="worker")

        if self.strategy is None:
            from flight.strategies.base import DefaultTrainerStrategy

            self.strategy = DefaultTrainerStrategy()

        # self._logger =

        try:
            self._device = torch.device(node.extra.get("device", "cpu"))
        except (AttributeError, TypeError):
            self._device = torch.device("cpu")

    def fit(
        self,
        node_state: WorkerState,
        model: TorchModule,
        data: TorchDataModule,
        validate_every_n_epochs: int = 1,
        ckpt_path: _PATH | None = None,
    ) -> list[Record]:
        """

        Args:
            node_state (WorkerState):
            model (TorchModule):
            data (TorchDataModule):
            validate_every_n_epochs:
            ckpt_path:

        Raises:
            - ValueError: Thrown when illegal values are given to arguments.

        Returns:

        """
        # TODO: Run the argument validation in a separate utility function to keep
        #  this function light.
        if validate_every_n_epochs < 1:
            raise ValueError("Illegal value for argument `validate_every_n_epochs`.")

        model.to(self._device)

        train_dataloader = data.train_data(self.node)
        valid_dataloader = data.valid_data(self.node)

        if not isinstance(train_dataloader, DataLoader):
            raise TypeError(
                "Method for argument `data.train_data(.)` must return a `DataLoader`."
            )
        if not isinstance(valid_dataloader, DataLoader | None):
            raise TypeError(
                "Method for argument `data.valid_data(.)` must return a `DataLoader` "
                "or `None`."
            )

        pbar_prefix = f"TorchTrainer(NodeID={self.node.idx})"
        if self._progress_bar:
            total = len(train_dataloader)
            if valid_dataloader is not None:
                total += len(valid_dataloader)
            total *= self.max_epochs
            pbar = tqdm.tqdm(total=total, desc=pbar_prefix)
        else:
            pbar = None

        optimizer = model.configure_optimizers()

        self._curr_step = 0
        self._results = []  # TODO: Convert to logger.

        for epoch in range(self.max_epochs):
            if pbar:
                pbar.set_description(f"{pbar_prefix} | {epoch=}")
            train_losses = self._epoch(
                epoch,
                node_state,
                model,
                optimizer,
                train_dataloader,
                pbar,
            )
            for loss in train_losses:
                self._results.append({"epoch": epoch, "train/loss": loss.item()})

            to_validate = all(
                [epoch % validate_every_n_epochs == 0, valid_dataloader is not None]
            )
            if to_validate:
                val_losses = self.validate(epoch, model, valid_dataloader, pbar)
                for loss in val_losses:
                    self._results.append(
                        {
                            "epoch": epoch,
                            "val/loss": loss.item(),
                            "step": self._curr_step,
                        }
                    )

        return self._results

    def _epoch(
        self,
        epoch: int,
        node_state: WorkerState,
        model: TorchModule,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        pbar: tqdm.tqdm | None,
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
            optimizer.step()

            self._results.append(
                {
                    "epoch": epoch,
                    "train/loss": loss.item(),
                    "train/batch_idx": batch_idx,
                    "train/step": self._curr_step,
                }
            )
            self._curr_step += 1

            if pbar is not None:
                pbar.update()

        return losses

    def validate(
        self,
        epoch: int,
        model: TorchModule,
        dataloader: DataLoader,
        pbar: tqdm.tqdm | None,
        *args,
        **kwargs,
    ):
        self._set_train_mode(model, False)

        losses = []
        for batch_idx, batch in enumerate(dataloader):
            batch = self._batch_to_device(batch)
            loss = model.validation_step(batch, batch_idx)

            self._results.append(
                {
                    "epoch": epoch,
                    "valid/loss": loss.item(),
                    "valid/batch_idx": batch_idx,
                    "valid/step": self._curr_step,
                }
            )

            if pbar is not None:
                pbar.update()

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

    @staticmethod
    def _set_train_mode(model: TorchModule, train_mode: bool = True) -> None:
        """
        Hidden utility function that switches the `TorchTrainer` to training or
        validation mode.

        Args:
            model (TorchModule): Model to set to training or evaluation mode.
            train_mode (bool): Training mode flag.
        """
        torch.set_grad_enabled(train_mode)
        if train_mode:
            model.train()
        else:
            model.eval()
