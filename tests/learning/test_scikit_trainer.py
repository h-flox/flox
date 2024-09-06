import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier

from flight.federation.topologies.node import Node, NodeKind
from flight.learning.metrics import InMemoryRecordLogger
from flight.learning.scikit import ScikitDataModule, ScikitModule, ScikitTrainer

# Seed for random number generation for data reusability
SEED = 9


@pytest.fixture
def node() -> Node:
    node = Node(idx=0, kind=NodeKind.WORKER)
    return node


@pytest.fixture
def data_scikit_cls() -> type[ScikitDataModule]:
    class MySciKitDataModule(ScikitDataModule):
        def __init__(self) -> None:
            super().__init__()
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
            return self.x_train, self.y_train

        def test_data(self, node: Node | None = None):
            return self.x_test, self.y_test

        def valid_data(self, node: Node | None = None):
            return self.x_val, self.y_val

    return MySciKitDataModule


@pytest.fixture
def data_classification_scikit() -> type[ScikitDataModule]:
    class MyClassificationDataModule(ScikitDataModule):
        def __init__(self):
            inputs, labels = make_classification(
                n_samples=10000, n_features=20, random_state=SEED
            )

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                inputs, labels, test_size=0.3
            )

        def train_data(self, node: Node | None = None):
            return self.x_train, self.y_train

        def test_data(self, node: Node | None = None):
            return self.x_test, self.y_test

        def valid_data(self, node: Node | None = None):
            return self.x_test, self.y_test

    return MyClassificationDataModule


class TestSciKitTrainer:
    def test_scikit_trainer(self, node, data_scikit_cls):
        """
        Tests a basic setup of using the `LightningTrainer` class for
        PyTorch-based models.
        """
        model_instance = MLPRegressor()
        model = ScikitModule(model_instance)
        data = data_scikit_cls()
        trainer = ScikitTrainer(node, partial=False)

        assert isinstance(model, ScikitModule)
        assert isinstance(trainer, ScikitTrainer)

        trainer.fit(model, data)

    def test_scikit_train(self, node, data_scikit_cls):
        """
        Tests that no errors occur during basic model fitting.
        """
        model_instance = MLPRegressor()
        model = ScikitModule(model_instance)
        data = data_scikit_cls()

        trainer = ScikitTrainer(node=node, partial=False)

        trainer.fit(model, data)
        assert True

    def test_scikit_test(self, node, data_scikit_cls):
        """
        Tests that no errors occur during basic model testing.
        """
        model_instance = MLPRegressor()
        model = ScikitModule(model_instance)
        data = data_scikit_cls()

        trainer = ScikitTrainer(node=node, partial=False)

        trainer.fit(model, data)

        result = trainer.test(model, data)
        assert isinstance(result, list)

    def test_scikit_val(self, node, data_scikit_cls):
        """
        Tests that no errors occur during basic model validation.
        """
        model_instance = MLPRegressor()
        model = ScikitModule(model_instance)
        data = data_scikit_cls()

        trainer = ScikitTrainer(
            node=node,
            partial=False,
            # log_every_n_steps=10
        )

        trainer.fit(model, data)

        records = trainer.validate(model, data)
        assert isinstance(records, list)

    def test_scikit_fit(self, node, data_classification_scikit):
        """
        Tests that a 'SciKitTrainable' can train a classifier on some
        classification dataset.
        """
        logger = InMemoryRecordLogger()
        model_instance = MLPClassifier()
        model = ScikitModule(model_instance)
        data = data_classification_scikit()

        trainer = ScikitTrainer(
            node=node,
            partial=True,
            # log_every_n_steps=5, logger=logger
        )

        trainer.fit(model, data)
        init_params = model.get_params()
        trainer.fit(model, data)
        fit_params = model.get_params()

        assert init_params is not fit_params
