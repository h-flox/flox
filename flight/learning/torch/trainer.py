from __future__ import annotations

import pathlib
import typing as t

import torch
import tqdm
from torch.utils.data import DataLoader

if t.TYPE_CHECKING:
    from flight.federation.topologies import Node
    from flight.federation.topologies.node import WorkerState
    from flight.strategies import TrainerStrategy
    from flight.types import Record

    from .data import TorchDataModule
    from .module import TorchModule

    _PATH: t.TypeAlias = pathlib.Path | str


class TorchTrainer:
    node: Node
    strategy: TrainerStrategy

    def __init__(
        self,
        max_epochs: int = 1,
        node: Node | None = None,
        strategy: TrainerStrategy | None = None,
        progress_bar: bool = True,
    ):
        self.max_epochs = max_epochs
        self._progress_bar = progress_bar
        self._curr_step = 0
        self._results: list[Record] = []

        # NOTE: We import within these `if` statements because Trainers can be
        # instantiated on remote devices where these modules may have not been
        # imported yet.
        if node is None:
            from flight.federation.topologies import Node, NodeKind

            self.node = Node(idx=0, kind=NodeKind.WORKER)
        else:
            self.node = node

        if strategy is None:
            from flight.strategies.base import DefaultTrainerStrategy

            self.strategy = DefaultTrainerStrategy()
        else:
            self.strategy = strategy

        # self._logger =
        self._device = self._get_device(self.node)

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
            - `ValueError`: Thrown when illegal values are given to arguments.

        Returns:
            A list of records containing the training results.
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

            # Validate the model during training.
            validate_now = epoch % validate_every_n_epochs == 0
            if validate_now and valid_dataloader is not None:
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
    ) -> list[torch.Tensor]:
        self._set_train_mode(model, True)

        losses: list[torch.Tensor] = []  # TODO: Check if we need to keep this.
        for batch_idx, batch in enumerate(dataloader):
            batch = self._batch_to_device(batch)
            loss = model.training_step(batch, batch_idx)

            # Perform backpropagation and call trainer strategy callbacks.
            optimizer.zero_grad()
            if self.strategy is not None:
                loss = self.strategy.before_backprop(node_state, loss)
            loss.backward()
            if self.strategy is not None:
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
            if loss is not None:
                self._results.append(
                    {
                        "epoch": epoch,
                        "valid/loss": loss.item(),
                        "valid/batch_idx": batch_idx,
                        "valid/step": self._curr_step,
                    }
                )
            else:
                pass

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

    @staticmethod
    def _get_device(node: Node) -> torch.device:
        """
        Gets the device on which the model will be trained.

        This method is used to determine the device on which the model will be trained.
        Specifically, it checks the `node.extra` attribute to see if a device is
        specified by the user there. If not, then 'cpu' will be used by default.

        Args:
            node (Node): The node on which the model will be trained.

        Returns:
            The device on which the model will be trained.

        Throws:
            - `TypeError`: Thrown when the `node.extra` attribute is neither `None` nor
                a `dict`.
        """
        match node.extra:
            case dict():
                try:
                    return torch.device(node.extra.get("device", "cpu"))
                except (AttributeError, TypeError):
                    return torch.device("cpu")
            case None:
                return torch.device("cpu")
            case _:
                raise TypeError("Node.extra must be a dictionary or None.")

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
