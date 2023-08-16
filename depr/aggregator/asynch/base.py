import lightning as L

from typing import Protocol, runtime_checkable, Iterable

from depr.typing import WorkerID
from depr.worker import WorkerLogicInterface


@runtime_checkable
class AsynchAggregatorLogicInterface(Protocol):
    def __init__(self):
        pass

    def on_module_broadcast(self):
        ...

    def on_worker_select(
        self, workers: dict[str, WorkerLogicInterface]
    ) -> Iterable[str]:
        ...

    def on_module_receive(self):
        ...

    def on_module_aggregate(
        self,
        global_module: L.LightningModule,
        worker_id: WorkerID,
        worker_update: L.LightningModule,
        **kwargs
    ):
        ...

    def on_module_evaluate(self, module: L.LightningModule):
        ...

    def stop_condition(self, state: dict) -> bool:
        ...


class AsynchAggregatorLogic:
    def __init__(self, **kwargs) -> None:
        pass

    def on_model_broadcast(self):
        pass

    def on_module_aggregate(
        self,
        global_module: L.LightningModule,
        worker_id: WorkerID,
        worker_module: L.LightningModule,
        **kwargs
    ):
        pass

    def on_module_eval(self, module: L.LightningModule):
        pass

    # def stop_condition(self, state: dict) -> bool:
    #     if state["curr_round"] < state["total_rounds"]:
    #         return False
    #     return True
