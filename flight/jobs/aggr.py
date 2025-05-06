from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flight.jobs.types import Result
    from flight.learning.module import Params, TorchModule
    from flight.system.topology import NodeID

AggrJobArgs: t.TypeAlias = t.Any


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
    return result
