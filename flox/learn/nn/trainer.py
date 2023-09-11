import datetime
from typing import Any, Optional

import pandas as pd
import torch

from collections import defaultdict
from torch.utils.data import DataLoader

from flox.flock.states import FloxWorkerState
from flox.learn.logger.csv import CSVLogger
from flox.learn.nn.model import FloxModule
from flox.strategies import Strategy


class Trainer:
    def __init__(self, logger: str = "csv", device="cpu"):
        self.device = device
        if logger == "csv":
            self.logger = CSVLogger()
        else:
            raise ValueError("Illegal value for `logger`.")

    def log(self):
        pass

    def fit(
        self,
        model: FloxModule,
        train_dataloader: DataLoader,
        num_epochs: int,
        node_state: Optional[FloxWorkerState] = None,
        strategy: Optional[Strategy] = None,
    ) -> pd.DataFrame:
        model.train()
        optimizer = model.configure_optimizers()
        self.logger.clear()

        torch.set_grad_enabled(True)
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(train_dataloader):
                loss = model.training_step(batch, batch_idx)
                optimizer.zero_grad()
                loss.backward()

                try:
                    strategy.wrk_on_after_train_step(node_state, loss)
                except NotImplementedError:
                    pass

                optimizer.step()

                # log data about training...
                self.logger.log_dict(
                    {
                        "train/loss": loss.item(),
                        "train/epoch": epoch,
                        "train/batch_idx": batch_idx,
                        "train/time": datetime.datetime.now(),
                    }
                )

        torch.set_grad_enabled(False)
        return self.logger.to_pandas()

    def test(self):
        pass
