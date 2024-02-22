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
    import pandas
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
    histories = []
    for res in results:
        match res.history:
            case pandas.DataFrame():
                histories.append(res.history)
            case dict():
                assert isinstance(res.history, dict)
                h = pandas.DataFrame.from_dict(res.history)
                histories.append(h)
            case _:
                raise ValueError

    history = pandas.concat(histories)
    return transfer.report(node_state, node.idx, node.kind, avg_state_dict, history)


def debug_aggregation_job(
    node: FlockNode, transfer: BaseTransfer, strategy: Strategy, results: list[Result]
) -> Result:
    import datetime
    import numpy
    import pandas

    result = next(iter(results))
    module = result.module
    node_state = dict(idx=node.idx)
    history = {
        "node/idx": [node.idx],
        "node/kind": [node.kind.to_str()],
        "train/loss": [numpy.nan],
        "train/epoch": [numpy.nan],
        "train/batch_idx": [numpy.nan],
        "train/time": [datetime.datetime.now()],
        "mode": "debug",
    }
    history = pandas.DataFrame.from_dict(history)
    return transfer.report(
        node_state, node.idx, node.kind, module.state_dict(), history
    )
