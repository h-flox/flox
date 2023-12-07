from __future__ import annotations

import datetime
from typing import Optional, Any

import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import DataLoader

from flox.flock import FlockNode
from flox.flock.states import FloxWorkerState
from flox.nn.logger import TensorboardLogger, CSVLogger
from flox.nn import FloxModule
from flox.strategies import Strategy


class Trainer:
    def __init__(
        self,
        node: FlockNode,
        logger: str = "csv",
        device="cpu",
        metadata: Optional[dict[str, Any]] = None,
        config: Optional[dict[str, Any]] = None,
    ):
        self.device = device
        self.config = config  # TODO: Not implemented to do anything at the moment.
        if logger == "csv":
            raise "not here"
            self.logger = CSVLogger(node, metadata)
        elif logger == "tensorboard":
            self.logger = TensorboardLogger(node, metadata)
        else:
            raise ValueError("Illegal value for `logger`.")

    def fit(
        self,
        model: FloxModule,
        train_dataloader: DataLoader,
        num_epochs: int,
        strategy: Optional[Strategy] = None,
        node_state: Optional[FloxWorkerState] = None,
        valid_dataloader: Optional[DataLoader] = None,
        valid_ckpt_path: Optional[Path | str] = None,
        round: Optional[int] = None,
    ) -> pd.DataFrame:
        model.train()
        optimizer = model.configure_optimizers()
        self.logger.clear()

        with torch.set_grad_enabled(True):
            for epoch in range(num_epochs):
                for batch_idx, batch in enumerate(train_dataloader):
                    loss = model.training_step(batch, batch_idx)
                    optimizer.zero_grad()
                    loss.backward()

                    try:
                        strategy.wrk_on_after_train_step(node_state, loss)
                    except NotImplementedError:
                        """
                        The current strategy does not override the `wrk_on_after_train_step()` callback.
                        """
                        pass

                    optimizer.step()

                    self.logger.log(
                        "train/loss",
                        loss.item(),
                        timestamp=datetime.datetime.now(),
                        round=round,
                        epoch=epoch,
                        batch_idx=batch_idx
                    )

                    # If a validation ``Dataset`` has been provided (i.e., the users
                    # have specified an object instance for it), then run validation.
                    if valid_dataloader is not None:
                        self.validate(model, valid_dataloader, epoch, valid_ckpt_path)

        return self.logger.to_pandas()

    def test(
        self,
        model: FloxModule,
        test_dataloader: DataLoader,
        ckpt_path: Optional[Path | str] = None,
    ):
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                model.test_step(batch, i)

    def validate(
        self,
        model: FloxModule,
        valid_dataloader: DataLoader,
        epoch: int,
        ckpt_path: Optional[Path | str] = None,
        round: Optional[int] = None,
    ):
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_dataloader):
                loss = model.validation_step(batch, batch_idx)
                self.logger.log(
                    "valid/loss",
                    loss.item(),
                    timestamp=datetime.datetime.now(),
                    round=round,
                    epoch=epoch,
                    batch_idx=batch_idx
                )
