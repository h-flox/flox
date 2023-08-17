import torch

from flox.aggregator.state import AggregatorState
from flox.flock import Flock, FlockNode
from flox.typing import StateDict
from typing import TypeAlias

from flox.worker.state import FloxWorkerState

Loss: TypeAlias = torch.Tensor


class Strategy:
    registry = {}

    def on_before_train_step(self, *args, **kwargs):
        pass

    def on_after_train_step(
        self, state: FloxWorkerState, loss: Loss, *args, **kwargs
    ) -> Loss:
        return loss

    def on_worker_selection(self, children: list[FlockNode], *args, **kwargs):
        return children

    def on_param_aggregation(self, state_dicts, *args, **kwargs):
        pass

    def on_before_aggr_send_params(self, params: StateDict, *args, **kwargs):
        return params

    def on_after_aggr_recv_params(self, *args, **kwargs) -> StateDict:
        pass

    def on_before_worker_send_params(self, *args, **kwargs) -> StateDict:
        pass

    def on_worker_recv_params(self, params: StateDict, *args, **kwargs):
        return params

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls.__name__.lower()] = cls
