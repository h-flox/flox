from __future__ import annotations

import typing as t
from dataclasses import dataclass

if t.TYPE_CHECKING:
    from flight.learning.module import Params, TorchModule
    from flight.system.topology import NodeID

    from ..strategy import Strategy
    from ..system.node import Node
    from .protocols import Result


@dataclass
class AggrJobArgs:
    """
    The arguments for the aggregator job.
    """

    node: Node
    child_results: list[Result]
    round_num: int
    handlers: list[t.Any]
    strategy: Strategy


class AggregatorJobProto(t.Protocol):
    @staticmethod
    def __call__(args: AggrJobArgs) -> Result:
        """
        This method is called when the AGGREGATOR job is launched.
        """


def aggregator_job(args: AggrJobArgs) -> Result:
    from flight.state import AggregatorState, WorkerState

    child_states: dict[NodeID, AggregatorState | WorkerState] = {}
    child_params: dict[NodeID, Params] = {}
    child_modules: dict[NodeID, TorchModule] = {}

    for result in args.child_results:
        idx = result.node.idx
        child_states[idx] = result.state
        child_params[idx] = result.params
        if isinstance(result.state, (AggregatorState, WorkerState)):
            child_modules[idx] = result.module
        else:
            raise TypeError(
                "Child state must be either `WorkerState` or `AggregatorState`."
            )

    aggr_state = AggregatorState()
    aggr_params = args.strategy.aggregate(child_params)

    return Result(
        node=args.node,
        state=aggr_state,
        params=aggr_params,
        extra={
            "child_states": child_states,
            "child_modules": child_modules,
            "round_num": args.round_num,
        },
    )
