from flox.flock import FlockNode
from flox.runtime.result import Result
from flox.runtime.transfer import BaseTransfer
from flox.strategies import Strategy


def aggregation_job(
    node: FlockNode, transfer: BaseTransfer, strategy: Strategy, results: list[Result]
) -> Result:
    """Aggregate the state dicts from each of the results.

    Args:
        node (FlockNode): The aggregator node.
        transfer (Transfer): ...
        strategy (Strategy): ...
        results (list[JobResult]): Results from children of ``node``.

    Returns:
        Aggregation results.
    """
    import pandas as pd
    from flox.flock.states import FloxAggregatorState

    child_states, child_state_dicts = {}, {}
    for result in results:
        idx = result.node_idx
        child_states[idx] = result.node_state
        child_state_dicts[idx] = result.state_dict

    node_state = FloxAggregatorState(node.idx)
    avg_state_dict = strategy.agg_param_aggregation(
        node_state, child_states, child_state_dicts
    )

    # As a note, this list comprehension is done to allow for `debug_mode` which uses a job on
    # worker nodes that returns empty dictionaries for its history object.
    histories = [
        res.history
        if isinstance(res.history, pd.DataFrame)
        else pd.DataFrame.from_dict(res.history)
        for res in results
    ]
    history = pd.concat(histories)
    return transfer.report(node_state, node.idx, node.kind, avg_state_dict, history)


def debug_aggregation_job(
    node: FlockNode, transfer: BaseTransfer, strategy: Strategy, results: list[Result]
) -> Result:
    result = next(iter(results))
    module = result.module
    node_state = dict(idx=node.idx)
    return transfer.report(node_state, node.idx, node.kind, module.state_dict(), {})
