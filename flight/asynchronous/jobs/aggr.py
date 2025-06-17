# Sana
from __future__ import annotations

import typing as t
from dataclasses import dataclass
import asyncio
from flight.jobs.protocols import Result

if t.TYPE_CHECKING:
    from flight.learning.module import TorchModule
    from flight.strategies.strategy import Strategy
    from flight.system.topology import NodeID

    from flight.learning.parameters import Params
    from flight.system.node import Node

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

async def aggregator_job(args: AggrJobArgs, result_event:asyncio.Event) -> Result:
    from flight.state import AggregatorState, WorkerState

    # Only aggregate the latest result (simulate "as soon as received")
    while not args.child_results:
        await result_event.wait()
        result_event.clear()

    # Use only the latest result for incremental aggregation
    latest_result = args.child_results[-1]
    idx = latest_result.node.idx

    # Build the state/params/modules dicts for all received so far
    child_states: dict[NodeID, AggregatorState | WorkerState] = {}
    child_params: dict[NodeID, 'Params'] = {}
    child_modules: dict[NodeID, 'TorchModule'] = {}

    for result in args.child_results:
        i = result.node.idx
        child_states[i] = result.state
        child_params[i] = result.params
        if isinstance(result.state, (AggregatorState, WorkerState)):
            child_modules[i] = result.module
        else:
            raise TypeError(
                "Child state must be either `WorkerState` or `AggregatorState`."
            )

    aggr_state = AggregatorState()
    # Aggregate using all received so far (or just the latest, if that's your policy)
    aggr_params = args.strategy.aggregate(child_params)

    return Result(
        node=args.node,
        state=aggr_state,
        params=aggr_params,
        extra={
            "child_states": child_states,
            "child_modules": child_modules,
            "round_num": args.round_num,
            "last_updated_worker": idx,
            "is_async_style": True,  # Mark this as "async-style" aggregation
        },
        module=None
    )