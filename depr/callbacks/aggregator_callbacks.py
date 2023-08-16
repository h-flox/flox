from typing import Protocol


class AggregatorCallbacks(Protocol):
    def on_module_broadcast(self):
        ...

    def on_worker_select(self):
        ...

    def on_module_gather(self):
        ...

    def on_module_aggregate(self):
        ...

    def on_module_test(self):
        ...

    def on_terminate(self):
        ...
