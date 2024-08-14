import lightning.pytorch as PL
import typing as t
import torch

from torch.utils.data import DataLoader

from ..modules.prototypes import DataLoadable
from ...federation.topologies.node import Node

from lightning.pytorch.loggers import Logger

_OUT_DICT: t.TypeAlias = t.Any
_PATH: t.TypeAlias = t.Any
_EVALUATE_OUTPUT: t.TypeAlias = t.Any

class LightningTrainer:
    def __init__(
        self,
        node: Node,
        max_epochs: int = 1,
        log_every_n_steps: int = 1,
        logger: Logger | None = None,
    ) -> None:
        self.node = node
        self.logger = logger
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps

        self.trainer = PL.Trainer(
            max_epochs=max_epochs,
            log_every_n_steps=log_every_n_steps,
            logger=self.logger
        )

        try:
            if node.extra:
                self._device = torch.device(node.extra.get("device", "cpu"))
            else:
                self._device = torch.device("cpu")
        except (AttributeError, TypeError):
            self._device = torch.device("cpu")

    def fit(
        self,
        model: PL.LightningModule,
        data: DataLoadable,
        ckpt_path: _PATH | None = None,
    ) -> _OUT_DICT | None:
        model.to(self._device)

        if self.max_epochs < 1:
            raise ValueError("Illegal value for argument 'max_epochs'.")

        if self.log_every_n_steps < 1:
            raise ValueError("Illegal value for argument 'log_every_n_steps'.")

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

        self.trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
            ckpt_path=ckpt_path,
        )

        if self.logger:
            return None
        else:
            return self.trainer.logged_metrics

    def test(
        self, model: PL.LightningModule, data: DataLoadable, ckpt_path: _PATH | None = None
    ) -> _EVALUATE_OUTPUT | None:
        test_dataloader = data.test_data(self.node)
        records = self.trainer.test(
            model=model, dataloaders=test_dataloader, ckpt_path=ckpt_path
        )

        if self.logger:
            return None
        else:
            return records

    def validate(
        self, model: PL.LightningModule, data: DataLoadable, ckpt_path: _PATH | None = None
    ) -> _EVALUATE_OUTPUT | None:
        valid_dataloader = data.valid_data(self.node)
        records = self.trainer.validate(
            model=model, dataloaders=valid_dataloader, ckpt_path=ckpt_path
        )
        
        if self.logger:
            return None
        else:
            return records