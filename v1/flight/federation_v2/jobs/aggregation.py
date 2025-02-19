from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from v1.flight.federation.types import AggrJobArgs, Result


def aggregation_job(args: AggrJobArgs) -> Result:
    from v1.flight.federation.records import broadcast_records
    from v1.flight.federation.types import Result
    from v1.flight.topologies.node import AggrState, WorkerState

    node = args.node
    child_results = args.child_results
    strategy = args.aggr_strategy
    transfer = args.transfer
    extra = {}

    # records = []
    # aggr_params = next(iter(child_results)).params
    # aggr_state = AggrState(node.idx, args.children)

    child_states = {}
    child_params = {}
    child_modules = {}
    for res in child_results:
        idx = res.node.idx
        child_states[idx] = res.node_state
        child_params[idx] = res.params
        if isinstance(res.node_state, (WorkerState, AggrState)):
            child_modules[idx] = res.module
        else:
            raise TypeError("Child state must be either `WorkerState` or `AggrState`.")

    aggr_state = AggrState(node.idx, children=args.children)
    aggr_params = strategy.aggregate_params(
        state=aggr_state,
        children_states=child_states,
        # children_params=child_params,
        children_modules=child_modules,
    )

    records = []
    for res in child_results:
        records.extend(res.records)

    broadcast_records(records, round=args.round_num)

    result = Result(
        node=node,
        node_state=aggr_state,
        params=aggr_params,
        records=records,
        extra=extra,
    )
    result = transfer.transfer(result)
    return result
