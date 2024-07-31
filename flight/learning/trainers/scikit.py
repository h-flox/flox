from flight.learning.datasets import DataLoadable
from flight.learning.modules.base import SciKitModule


class ScikitTrainer:
    def __init__(self, partial: bool = True):
        self.partial = partial

    def fit(self, model: SciKitModule, data: DataLoadable):
        inputs, targets = data.load()
        model.fit(inputs, targets)

    def test(self):
        pass

    def validate(self):
        pass
