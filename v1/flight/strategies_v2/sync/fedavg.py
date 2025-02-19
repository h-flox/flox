from v1.flight.federation_v2.events import CoordEvent, on
from v1.flight.strategies import Strategy


class FedAvg(Strategy):
    def __init__(self):
        pass

    def aggregation_policy(self, *args, **kwargs):
        _ = self.__name__
        return 0

    def selection_policy(self, *args, **kwargs):
        _ = self.__name__
        return 1

    @on(CoordEvent.ROUND_START)
    def my_round_state(self, context):
        print("Hello!")


class FedProx(FedAvg):
    @on(...)
    def before_backprop(self):
        pass


class MyStrategy(FedAvg, StandardCoordinatorValidation):
    pass
