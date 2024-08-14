import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from flight.federation.topologies.node import Node, NodeKind
from flight.learning.trainers.scikit import ScikitTrainer
from flight.learning.modules.prototypes import DataLoadable
from flight.learning.modules.scikit import ScikitTrainable

@pytest.fixture
def node() -> Node:
    node = Node(idx=0, kind=NodeKind.WORKER)
    return node

@pytest.fixture
def data_scikit_cls() -> type[DataLoadable]:
    class MySciKitDataModule:
        def __init__(self) -> None:
            self.num_samples = 10_000
            self.num_features = 1

            self.x = np.random.randn(self.num_samples, self.num_features)
            self.y = np.random.randn(self.num_samples)

            self.x_train, self.x_temp, self.y_train, self.y_temp = train_test_split(
            self.x, self.y, test_size=0.4, random_state=42
            )
            self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(
            self.x_temp, self.y_temp, test_size=0.5, random_state=42
            )

        def train_data(self, node: Node | None = None):
            return (self.x_train, self.y_train)
        
        def test_data(self, node: Node | None = None):
            return (self.x_test, self.y_test)
        
        def valid_data(self, node: Node | None = None):
            return (self.x_val, self.y_val)
        
    return MySciKitDataModule

class TestSciKitTrainer:
    def test_scikit_trainer(self, node, data_scikit_cls):
        """
        Tests a basic setup of using the `LightningTrainer` class for PyTorch-based models.
        """
        model_instance = MLPRegressor()
        model = ScikitTrainable(model_instance)
        data = data_scikit_cls()
        trainer = ScikitTrainer(node, False)

        assert isinstance(model, ScikitTrainable)
        assert isinstance(trainer, ScikitTrainer)

        trainer.fit(model, data)
        #assert isinstance(results, dict)

    def test_scikit_train(self, node, data_scikit_cls):
        model_instance = MLPRegressor()
        model = ScikitTrainable(model_instance)
        data = data_scikit_cls()

        trainer = ScikitTrainer(
            node=node, 
            partial=False, 
            log_every_n_steps=10
        )

        trainer.fit(model, data)
        assert True

    def test_scikit_test(self, node, data_scikit_cls):
        model_instance = MLPRegressor()
        model = ScikitTrainable(model_instance)
        data = data_scikit_cls()

        trainer = ScikitTrainer(
            node=node,
            partial=False,
            log_every_n_steps=10
        )

        trainer.fit(model, data)

        records = trainer.test(model, data)
        assert isinstance(records, float)
    
    def test_scikit_val(self, node, data_scikit_cls):
        model_instance = MLPRegressor()
        model = ScikitTrainable(model_instance)
        data = data_scikit_cls()

        trainer = ScikitTrainer(
            node=node,
            partial=False,
            log_every_n_steps=10
        )

        trainer.fit(model,data)

        records = trainer.validate(model, data)
        assert isinstance(records, list)
