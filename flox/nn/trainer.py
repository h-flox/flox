import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from flox.flock.states import WorkerState
from flox.nn import FloxModule
from flox.nn.logger.csv import CSVLogger
from flox.strategies import TrainerStrategy


class Trainer:
    def __init__(
        self,
        logger: str = "csv",
        device="cpu",
        config: dict[str, Any] | None = None,
    ):
        self.device = device
        self.config = config  # TODO: Not implemented to do anything at the moment.
        if logger == "csv":
            self.logger = CSVLogger()
        else:
            raise ValueError("Illegal value for `logger`.")

    def fit(
        self,
        model: FloxModule,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        num_epochs: int,
        trainer_strategy: TrainerStrategy,
        node_state: WorkerState,
        valid_dataloader: DataLoader | None = None,
        valid_ckpt_path: Path | str | None = None,
    ) -> pd.DataFrame:
        model.train()
        self.logger.clear()

        with torch.set_grad_enabled(True):
            for epoch in range(num_epochs):
                for batch_idx, batch in enumerate(train_dataloader):
                    loss = model.training_step(batch, batch_idx)
                    loss = trainer_strategy.before_backprop(loss)

                    optimizer.zero_grad()
                    loss.backward()
                    loss = trainer_strategy.after_backprop(loss)
                    optimizer.step()

                    # log data about training
                    rec = {
                        "train/loss": loss.item(),
                        "train/epoch": epoch,
                        "train/batch_idx": batch_idx,
                        "train/time": datetime.datetime.now(),
                    }
                    self.logger.log_dict(rec)

                    # If a validation ``Dataset`` has been provided (i.e., the users
                    # have specified an object instance for it), then run validation.
                    if valid_dataloader is not None:
                        self.validate(model, valid_dataloader, epoch, valid_ckpt_path)

        return self.logger.to_pandas()

    def test(
        self,
        model: FloxModule,
        test_dataloader: DataLoader,
        ckpt_path: Path | str | None = None,
    ):
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                model.test_step(batch, i)

    def validate(
        self,
        model: FloxModule,
        valid_dataloader: DataLoader,
        epoch: int,
        ckpt_path: Path | str | None = None,
    ):
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_dataloader):
                loss = model.validation_step(batch, batch_idx)
                self.logger.log_dict(
                    {
                        "valid/loss": loss.item(),
                        "valid/epoch": epoch,
                        "valid/batch_idx": batch_idx,
                        "valid/time": datetime.datetime.now(),
                    }
                )
