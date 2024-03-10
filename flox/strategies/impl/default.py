from flox.strategies.aggregator import AggregatorStrategy
from flox.strategies.client import ClientStrategy
from flox.strategies.trainer import TrainerStrategy
from flox.strategies.worker import WorkerStrategy


class DefaultClientStrategy(ClientStrategy):
    def __init__(self):
        super().__init__()


class DefaultAggregatorStrategy(AggregatorStrategy):
    def __init__(self):
        super().__init__()


class DefaultWorkerStrategy(WorkerStrategy):
    def __init__(self):
        super().__init__()


class DefaultTrainerStrategy(TrainerStrategy):
    def __init__(self):
        super().__init__()
