import datetime
from typing import Optional

import pandas as pd
import torch

from torch.utils.data import DataLoader

from flox.flock.states import FloxWorkerState
from flox.nn.logger.csv import CSVLogger
from flox.nn import FloxModule
from flox.strategies import Strategy


class Trainer:
    def __init__(
        self, logger: str = "csv", device="cpu", config: Optional[dict] = None
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
        train_dataloader: DataLoader,
        num_epochs: int,
        node_state: Optional[FloxWorkerState] = None,
        strategy: Optional[Strategy] = None,
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

                    optimizer.step()

                    # log data about training
                    self.logger.log_dict(
                        {
                            "train/loss": loss.item(),
                            "train/epoch": epoch,
                            "train/batch_idx": batch_idx,
                            "train/time": datetime.datetime.now(),
                        }
                    )

        return self.logger.to_pandas()

    def test(self):
        pass

    def validate(self):
        pass
