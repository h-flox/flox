from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from .types import AggrJobArgs, Result


def default_aggr_job(args: AggrJobArgs) -> Result:
    from flight.federation.jobs.types import Result
    from flight.federation.records import broadcast_records
    from flight.federation.topologies.node import AggrState

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
    for res in child_results:
        idx = res.node.idx
        child_states[idx] = res.node_state
        child_params[idx] = res.params

    aggr_state = AggrState(node.idx, children=args.children)
    aggr_params = strategy.aggregate_params(
        state=aggr_state,
        children_states=child_states,
        children_params=child_params,
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
    result = transfer(result)
    return result
