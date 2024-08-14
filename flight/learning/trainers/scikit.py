import typing as t

from flight.learning.modules.prototypes import DataLoadable, SciKitModule
from flight.learning.modules.scikit import ScikitTrainable
from flight.federation.topologies.node import Node
from flight.learning.metrics import RecordLogger

#log_los -> MLPClassifier | mean_squared_error -> MLPRegressor
from sklearn.metrics import log_loss, mean_squared_error # type: ignore
from sklearn.neural_network import MLPClassifier, MLPRegressor # type: ignore

import numpy as np

class ScikitTrainer:
    def __init__(self, node: Node, partial: bool = True, log_every_n_steps: int = 1, logger: RecordLogger | None = None):
        self.partial = partial
        self.initalizing = True
        self.node = node
        self.logger = logger
        self.log_every_n_steps = log_every_n_steps

    def fit(self, model: ScikitTrainable, data: DataLoadable):
        inputs, targets = data.train_data(self.node)

        if len(inputs) != len(targets):
            raise ValueError(f"Number of 'inputs'({len(inputs)}) does not match number of 'targets'({len(targets)}).")
        
        if self.partial:
            if self.initalizing:
                # Initalizing the model with the unique classes on the first call
                classes = np.unique(targets)
                model.module.partial_fit(inputs, targets, classes)
                self.initalizing = False
            else:
                # Subsequent 'partial_fit' calls
                model.module.partial_fit(inputs, targets)
        else:
            model.module.fit(inputs, targets)

    def test(self, model: ScikitTrainable, data: DataLoadable):
        dataset = data.test_data(self.node)
        if not dataset is None:
            inputs, targets = dataset
        else:
            raise AttributeError("'DataLoadable' object does not support test data.")

        if len(inputs) != len(targets):
            raise ValueError(f"Number of 'inputs'({len(inputs)}) does not match number of 'targets'({len(targets)}).")
        
        test_acc = model.module.score(inputs, targets)

        if self.logger:
            record: t.Mapping = {'test/acc': test_acc}
            self.logger.log(**record)
        else:
            return test_acc

    def validate(self, model: ScikitTrainable, data: DataLoadable):
        dataset = data.valid_data(self.node)
        if not dataset is None:
            inputs, targets = dataset
        else:
            raise AttributeError("'DataLoadable' does not support validation data.")

        if len(inputs) != len(targets):
            raise ValueError(f"Number of 'inputs'({len(inputs)}) does not match number of 'targets'({len(targets)}).")
        
        records = []
        running_loss = 0.0
        step = 0

        for x, target in zip(inputs,targets):
            step += 1
            pred = model.module.predict(x.reshape(1,1))

            if model.module is MLPClassifier:
                running_loss += log_loss(target, pred)
            elif model.module is MLPRegressor:
                running_loss += mean_squared_error(target, pred)

            if step % self.log_every_n_steps == 0:
                record: t.Mapping = {'val/loss': running_loss/self.log_every_n_steps}
                if self.logger:
                    self.logger.log(**record)
                else:
                    records.append(record)
                running_loss = 0.0
        
        if not self.logger:
            return records
        