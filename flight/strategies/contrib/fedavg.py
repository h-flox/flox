from flight.strategies.strategy import Strategy


class FedAvg(Strategy):
    def __init__(self):
        super().__init__()

    def aggregation_policy(self, *args, **kwargs):
        # TODO
        pass

    def selection_policy(self, *args, **kwargs):
        # TODO
        pass
