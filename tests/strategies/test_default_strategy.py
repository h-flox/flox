from flight.strategies.base import DefaultTrainerStrategy


def test_default_trainer_strategy():
    strategy = DefaultTrainerStrategy()
    assert len(strategy.hparams()) == 0
