from flight.jobs import worker_job
from flight.strategy import Strategy
from flight.events import *


def foo(*args, **kwargs):
    pass

class TestStrategy(Strategy):
    def __init__(self):
        super().__init__(foo, foo)

    @on(WorkerEvents.STARTED)
    def on_start(self, context):
        pass

    @on(WorkerEvents.COMPLETED)
    def on_completion(self, context):
        pass
