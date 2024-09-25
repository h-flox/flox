from sklearn.neural_network import MLPRegressor, MLPClassifier

from flight.learning.metrics import InMemoryRecordLogger
from flight.learning.scikit import ScikitModule, ScikitTrainer
from testing.fixtures import (
    worker_node,
    scikit_regr_data_module,
    scikit_clf_data_module,
)


def test_scikit_trainer(worker_node, scikit_regr_data_module):
    """
    Tests a basic setup of using the `LightningTrainer` class for
    PyTorch-based models.
    """
    model_instance = MLPRegressor()
    model = ScikitModule(model_instance)
    trainer = ScikitTrainer(worker_node, partial=False)

    assert isinstance(model, ScikitModule)
    assert isinstance(trainer, ScikitTrainer)

    trainer.fit(model, scikit_regr_data_module)


def test_scikit_train(worker_node, scikit_regr_data_module):
    """
    Tests that no errors occur during basic model fitting.
    """
    model_instance = MLPRegressor()
    model = ScikitModule(model_instance)

    trainer = ScikitTrainer(node=worker_node, partial=False)

    trainer.fit(model, scikit_regr_data_module)
    assert True


def test_scikit_test(worker_node, scikit_regr_data_module):
    """
    Tests that no errors occur during basic model testing.
    """
    model_instance = MLPRegressor()
    model = ScikitModule(model_instance)

    trainer = ScikitTrainer(node=worker_node, partial=False)

    trainer.fit(model, scikit_regr_data_module)

    result = trainer.test(model, scikit_regr_data_module)
    assert isinstance(result, list)


def test_scikit_val(worker_node, scikit_regr_data_module):
    """
    Tests that no errors occur during basic model validation.
    """
    model_instance = MLPRegressor()
    model = ScikitModule(model_instance)

    trainer = ScikitTrainer(
        node=worker_node,
        partial=False,
        # log_every_n_steps=10
    )

    trainer.fit(model, scikit_regr_data_module)

    records = trainer.validate(model, scikit_regr_data_module)
    assert isinstance(records, list)


def test_scikit_fit(worker_node, scikit_clf_data_module):
    """
    Tests that a 'SciKitTrainable' can train a classifier on some
    classification dataset.
    """
    logger = InMemoryRecordLogger()
    model_instance = MLPClassifier()
    model = ScikitModule(model_instance)

    trainer = ScikitTrainer(
        node=worker_node,
        partial=True,
        # log_every_n_steps=5, logger=logger
    )

    trainer.fit(model, scikit_clf_data_module)
    init_params = model.get_params()
    trainer.fit(model, scikit_clf_data_module)
    fit_params = model.get_params()

    assert init_params is not fit_params
