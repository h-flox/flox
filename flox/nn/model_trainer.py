import datetime
from pathlib import Path

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
        trainer_strategy: TrainerStrategy,
        log_every_n_batches: int = 1,  # 10,
        device=torch.device("cpu"),
    ):
        """
        Note:
            The `log_every_n_batches` parameter is very sensitive. If you set it too large, you may not record any
            data from fitting and result in errors when trying to merge the different histories across workers.
            Also, if the value is too small (e.g., `log_every_n_batches=1`), then your output files will become very large.
        """
        self.trainer_strategy = trainer_strategy
        self.logger = CSVLogger()
        self.log_every_n_batches = log_every_n_batches
        self.device = device

        # self.device = "cpu"
        # if torch.cuda.is_available():
        #     self.device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        # else:
        #     self.device = torch.device("cpu")

    def fit(
        self,
        model: FloxModule,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        num_epochs: int,
        node_state: WorkerState,
        valid_dataloader: DataLoader | None = None,
        valid_ckpt_path: Path | str | None = None,
    ) -> pd.DataFrame:
        self.logger.clear()
        model.to(self.device)
        with torch.set_grad_enabled(True):
            for epoch in range(num_epochs):
                avg_loss = self._epoch(
                    epoch,
                    model,
                    node_state,
                    optimizer,
                    train_dataloader,
                    valid_ckpt_path,
                    valid_dataloader,
                )
                # rec = {
                #     "train/loss": avg_loss,
                #     "train/epoch": epoch,
                #     "train/batch_idx": None,
                #     "train/time": datetime.datetime.now(),
                # }
                # self.logger.log_dict(rec)

                # If a validation ``Dataset`` has been provided (i.e., the users
                # have specified an object instance for it), then run validation.
                if valid_dataloader is not None:
                    self.validate(model, valid_dataloader, epoch, valid_ckpt_path)

        # model.to("cpu")
        return self.logger.to_pandas()

    def test(
        self,
        model: FloxModule,
        test_dataloader: DataLoader,
        ckpt_path: Path | str | None = None,
    ):
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                model.test_step(batch, batch_idx)

    def validate(
        self,
        model: FloxModule,
        valid_dataloader: DataLoader,
        epoch: int,
        ckpt_path: Path | str | None = None,
    ):
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_dataloader):
                loss = model.validation_step(batch, batch_idx)
                # self.logger.log_dict(
                #     {
                #         "valid/loss": loss.item(),
                #         "valid/epoch": epoch,
                #         "valid/batch_idx": batch_idx,
                #         "valid/time": datetime.datetime.now(),
                #     }
                # )

    def _epoch(
        self,
        epoch_index: int,
        model: FloxModule,
        node_state: WorkerState,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        valid_ckpt_path: Path | str | None = None,
        valid_dataloader: DataLoader | None = None,
    ):
        def log_condition(batch_idx: int):
            conditions = [
                batch_idx % self.log_every_n_batches == self.log_every_n_batches - 1,
                batch_idx == len(train_dataloader),
            ]
            return any(conditions)

        model.train(True)
        # model.to(self.device)
        running_acc = 0.0
        total_loss = 0.0
        running_loss = 0.0
        last_loss = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx)

            # log data about training
            running_acc += model.last_accuracy.item()
            running_loss += loss.item()
            total_loss += loss.item()

            loss = self.trainer_strategy.before_backprop(node_state, loss)
            loss.backward()
            loss = self.trainer_strategy.after_backprop(node_state, loss)
            optimizer.step()

            if log_condition(batch_idx):
                running_loss /= self.log_every_n_batches
                running_acc /= self.log_every_n_batches
                self.logger.log_dict(
                    {
                        "train/acc": running_acc,
                        "train/loss": running_loss,
                        "train/epoch": epoch_index,
                        "train/batch_idx": batch_idx,
                        "train/time": datetime.datetime.now(),
                    }
                )
                running_acc, running_loss = 0.0, 0.0

        if total_loss / len(train_dataloader) > last_loss:
            return total_loss / len(train_dataloader)
        else:
            return last_loss
