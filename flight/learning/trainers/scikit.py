import typing as t

import numpy as np

# log_los -> MLPClassifier | mean_squared_error -> MLPRegressor
from sklearn.metrics import log_loss, mean_squared_error  # type: ignore
from sklearn.neural_network import MLPClassifier, MLPRegressor  # type: ignore

from flight.federation.topologies.node import Node
from flight.learning.metrics import RecordLogger
from flight.learning.modules.prototypes import DataModuleProto
from flight.learning.modules.scikit import ScikitTrainable


class ScikitTrainer:
    """
    This is a trainer object that is responsible for training, testing, and
    validating models implemented in Scikit-Learn.
    """

    def __init__(
        self,
        node: Node,
        partial: bool = True,
        log_every_n_steps: int = 1,
        logger: RecordLogger | None = None,
        loss_fn: t.Callable[..., float] | None = None,
    ):
        """

        Args:
            node (Node): The node that training is occurring on.
            partial (bool, optional): If True the Trainer object will perform a
                partial fit. Defaults to True.
            log_every_n_steps (int, optional): How often to log within steps.
                Defaults to 1.
            logger (RecordLogger | None, optional): Optional logging device to capture
                learning metrics. Defaults to None.
            loss_fn (t.Callable[..., float] | None, optional): The function to be used
                to calculate loss. Defaults to None.
        """
        self.partial = partial
        self._first_partial_fit = True
        self.node = node
        self.logger = logger
        self.log_every_n_steps = log_every_n_steps
        self.loss_fn = loss_fn

    def fit(self, model: ScikitTrainable, data: DataModuleProto) -> None:
        """
        Runs full or partial optimization routine on a model using
        training data from 'data'.

        Args:
            model (ScikitTrainable): Trainable object that will be used to train a
                module on the given dataset.
            data (DataModuleProto): Data object that will provide training data
                necessary for the fitting process.

        Throws:
            - `ValueError`: Thrown if an unequal number of inputs and labels are
                given from 'data'.
        """
        self._infer_loss_fn(model)

        inputs, targets = data.train_data(self.node)

        if len(inputs) != len(targets):
            raise ValueError(
                f"Number of 'inputs'({len(inputs)}) does not match number of "
                f"'targets'({len(targets)})."
            )

        if self.partial:
            if self._first_partial_fit:
                # Initalizing the model with the unique classes on the first call
                classes = np.unique(targets)
                model.module.partial_fit(inputs, targets, classes)
                self._first_partial_fit = False
            else:
                # Subsequent 'partial_fit' calls
                model.module.partial_fit(inputs, targets)
        else:
            model.module.fit(inputs, targets)

    def test(self, model: ScikitTrainable, data: DataModuleProto) -> float:
        """
        Performs one evaluation epoch on the already trained model,
        finding model accuracy.

        Args:
            model (ScikitTrainable): The trained module that will be tested on a
                set of data.
            data (DataModuleProto): Data object that will provide test data necessary
                for the testing process.

        Throws:
            - `AttributeError`: Thrown if the 'data.test_data' returns None.
            - `ValueError`: Thrown if an unequal number of inputs and labels are given
                from 'data'.

        Returns:
            float: The mean accuracy on the given test data.
        """
        dataset = data.test_data(self.node)
        if dataset is not None:
            inputs, targets = dataset
        else:
            raise AttributeError("'DataLoadable' object does not support test data.")

        if len(inputs) != len(targets):
            raise ValueError(
                f"Number of 'inputs'({len(inputs)}) does not match number of "
                f"'targets'({len(targets)})."
            )

        test_acc = model.module.score(inputs, targets)

        if self.logger:
            record: t.Mapping = {"test/acc": test_acc}
            self.logger.log(**record)

        return test_acc

    def validate(
        self, model: ScikitTrainable, data: DataModuleProto
    ) -> t.Mapping[str, float]:
        """
        Performs one evaluation epoch on a given model to find validation loss.

        Args:
            model (ScikitTrainable): The module that will be validated on a set of data.
            data (DataModuleProto): Data object that will provide validation data
                necessary for the validation process.

        Throws:
            - `AttributeError`: Thrown if the 'data.valid_data' returns None.
            - `ValueError`: Thrown if an unequal number of inputs and labels are given
                from 'data'.

        Returns:
            t.Mapping[str, float]: List containing the metrics sent to loggers.
        """
        dataset = data.valid_data(self.node)
        if dataset is not None:
            inputs, targets = dataset
        else:
            raise AttributeError("'DataLoadable' does not support validation data.")

        if len(inputs) != len(targets):
            raise ValueError(
                f"Number of 'inputs'({len(inputs)}) does not match number of "
                f"'targets'({len(targets)})."
            )

        pred = model.module.predict(inputs)

        loss = 0.0
        if isinstance(model.module, MLPClassifier):
            loss = log_loss(targets, pred)
        elif isinstance(model.module, MLPRegressor):
            loss = mean_squared_error(targets, pred)

        record: t.Mapping = {"val/loss": loss}
        if self.logger:
            self.logger.log(**record)

        return record

    def _infer_loss_fn(self, model: ScikitTrainable):
        if self.loss_fn is not None:
            return
        elif isinstance(model.module, MLPClassifier):
            self.loss_fn = log_loss
        elif isinstance(model.module, MLPRegressor):
            self.loss_fn = mean_squared_error
