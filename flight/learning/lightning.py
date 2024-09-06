import typing as t
import warnings

import lightning.pytorch as PL
from lightning_utilities.core.rank_zero import log as device_logger
from torch.utils.data import DataLoader

from flight.federation.topologies.node import Node

_OUT_DICT: t.TypeAlias = t.Any
_PATH: t.TypeAlias = t.Any
_EVALUATE_OUTPUT: t.TypeAlias = t.Any

# TODO: Find a way to incorporate a node into each LightningDataModule,
#  similarly to how a 'DataLoadable' does.


class LightningTrainer:
    """
    This is a trainer object that is responsible for the training, testing, and
    validation processes. This object is a wrapper class for PyTorch Lightning's
    ['Trainer'](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
    that makes interacting with a Lightning Trainer easier within Flight.

    Notes:
        This class relies on the `LightningDataModule` provided by the PyTorch
        Lightning framework. Except the local training done by Flight will assume
        that there is a hidden attribute, `_node` which is used to fetch the train,
        validation, and test data loaders.
    """

    def __init__(self, node: Node, **kwargs) -> None:
        """

        Args:
            node (Node): The node that training is occurring on.
            **kwargs (dict[str, t.Any]): Key word arguments that can be used to
                customize the PyTorch Lighting trainer.
        """
        self.node = node
        if "logger" in kwargs.keys():
            self._pl_logger = True
        else:
            self._pl_logger = False

        device_logger.disabled = True
        # TODO: There are still issues with silencing deprecation warnings.
        #       These will need to be addressed.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.trainer = PL.Trainer(
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                **kwargs,
            )

    def fit(
        self,
        model: PL.LightningModule,
        data: PL.LightningDataModule,
        ckpt_path: _PATH | None = None,
    ) -> _OUT_DICT | None:
        """
        Runs optimization routine on the PyTorch Lightning Trainer, fitting 'model' to
        the training dataset.

        Args:
            model (PL.LightningModule): PyTorch Lightning module that will be trained
                on a 'LightningDataModule' object's training data.
            data (PL.LightningDataModule): The data object that provides the training
                and validation.
            ckpt_path (_PATH | None, optional): Path/URL of the checkpoint from which
                training is resumed. Defaults to None.

        Raises:
            - TypeError: Thrown when training or validation data is not held in a
                'DataLoader' object.

        Returns:
            _OUT_DICT | None: A dictionary containing the metrics sent to loggers if no
                logger is present. Returns None if a logger is present.
        """
        train_dataloader = data.train_dataloader()
        valid_dataloader = data.val_dataloader()

        if not isinstance(train_dataloader, DataLoader):
            raise TypeError(
                "Method for argument `data.train_data(.)` must return a `DataLoader`."
            )

        if not isinstance(valid_dataloader, DataLoader | None):
            raise TypeError(
                "Method for argument `data.valid_data(.)` must return a `DataLoader` "
                "or `None`."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.trainer.fit(
                model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader,
                ckpt_path=ckpt_path,
            )

        if self._pl_logger:
            return None
        else:
            return self.trainer.logged_metrics

    def test(
        self,
        model: PL.LightningModule,
        data: PL.LightningDataModule,
        ckpt_path: _PATH | None = None,
    ) -> _EVALUATE_OUTPUT | None:
        """
        Performs one evaluation epoch on the already fitted model with
        testing data from 'data'.

        Args:
            model (PL.LightningModule): PyTorch Lightning module that will be tested on
                a 'LightningDataModule' object's test data.
            data (PL.LightningDataModule): The data object that provides the testing
                data.
            ckpt_path (_PATH | None, optional): Path/URL of the checkpoint from
                which training is resumed. Defaults to None.

        Returns:
            _EVALUATE_OUTPUT | None: List of dictionaries with metrics logged during
                the test phase if no logger is present. Returns None if a
                logger is present.
        """
        dataloader = data.test_dataloader()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            records = self.trainer.test(
                model=model, dataloaders=dataloader, ckpt_path=ckpt_path
            )

        if self._pl_logger:
            return None
        else:
            return records

    def validate(
        self,
        model: PL.LightningModule,
        data: PL.LightningDataModule,
        ckpt_path: _PATH | None = None,
    ) -> _EVALUATE_OUTPUT | None:
        """Performs one evaluation epoch on the model with validation data from 'data'.

        Args:
            model (PL.LightningModule): PyTorch Lightning module that will be
                validated on a 'LightningDataModule' object's validation data.
            data (PL.LightningModule): The data object that provides the
                validation data.
            ckpt_path (_PATH | None, optional): Path/URL of the checkpoint from
                which training is resumed. Defaults to None.

        Returns:
            _EVALUATE_OUTPUT | None: List of dictionaries with metrics logged
                during the test phase if no logger is present.
                Returns `None` if a logger is present.
        """
        dataloader = data.val_dataloader()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            records = self.trainer.validate(
                model=model, dataloaders=dataloader, ckpt_path=ckpt_path
            )

        if self._pl_logger:
            return None
        else:
            return records
