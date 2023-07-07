import lightning as L

from typing import Protocol, runtime_checkable, Iterable

from flox.worker import WorkerLogicInterface


@runtime_checkable
class AggregatorLogicInterface(Protocol):
    def __init__(self):
        pass

    def on_module_broadcast(self):
        ...

    def on_worker_select(self, workers: dict[str, WorkerLogicInterface]) -> Iterable[str]:
        ...

    def on_module_receive(self):
        ...

    def on_module_aggregate(
            self,
            module: L.LightningModule,
            workers: dict[str, WorkerLogicInterface],
            updates: dict[str, L.LightningModule],
            **kwargs
    ):
        ...

    def on_module_evaluate(self, module: L.LightningModule):
        ...

    def stop_condition(self, state: dict) -> bool:
        ...


class SimpleAggregatorLogic:
    def __init__(self, **kwargs) -> None:
        pass

    def on_model_broadcast(self):
        pass

    def on_module_aggregate(
            self,
            module: L.LightningModule,
            workers: dict[str, WorkerLogicInterface],
            updates: dict[str, L.LightningModule],
            **kwargs
    ):
        pass

    def on_module_eval(self, module: L.LightningModule):
        pass

    def stop_condition(self, state: dict) -> bool:
        if state["curr_round"] < state["total_rounds"]:
            return False
        return True
