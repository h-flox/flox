import torch

from flox.aggregator.state import AggregatorState
from flox.flock import Flock, FlockNode, FlockNodeID
from flox.typing import StateDict
from typing import TypeAlias

from flox.flock.states import FloxWorkerState

Loss: TypeAlias = torch.Tensor


class Strategy:
    """Base class for the logical blocks of a FL process.

    A ``Strategy`` in FLoX is used to implement the logic of an FL process. A ``Strategy`` provides
    a number of callbacks which can be overridden to inject pieces of logic throughout the FL process.
    Some of these callbacks are run on the aggregator nodes while others are run on the worker nodes.
    """

    registry = {}

    def agg_on_param_aggregation(
        self,
        states: dict[FlockNodeID, FloxWorkerState],
        state_dicts: dict[FlockNodeID, StateDict],
        *args,
        **kwargs,
    ):
        pass

    def agg_on_before_submit_params(self, params: StateDict, *args, **kwargs):
        return params

    def agg_on_after_collect_params(self, *args, **kwargs) -> StateDict:
        pass

    def agg_on_worker_selection(self, children: list[FlockNode], *args, **kwargs):
        return children

    def wrk_on_before_train_step(self, *args, **kwargs):
        pass

    def wrk_on_after_train_step(
        self, state: FloxWorkerState, loss: Loss, *args, **kwargs
    ) -> Loss:
        return loss

    def wrk_on_before_submit_params(self, *args, **kwargs) -> StateDict:
        pass

    def wrk_on_recv_params(self, params: StateDict, *args, **kwargs):
        return params

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls.__name__.lower()] = cls

    @classmethod
    def get_strategy(cls, name: str):
        name = name.lower()
        if name in cls.registry:
            return cls.registry[name]
        else:
            raise KeyError(f"Strategy name ({name=}) is not in the Strategy registry.")
