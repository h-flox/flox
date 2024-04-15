import typing as t

from flox.flock import FlockNode, NodeID
from flox.jobs.protocols import AggregableJob
from flox.runtime.result import Result
from flox.runtime.transfer import BaseTransfer
from flox.strategies import AggregatorStrategy


class AggregateJob(AggregableJob):
    @staticmethod
    def __call__(
        node: FlockNode,
        children: t.Iterable[FlockNode],
        transfer: BaseTransfer,
        aggr_strategy: AggregatorStrategy,
        results: list[Result],
    ) -> Result:
        """Aggregate the state dicts from each of the results.

        Args:
            node (FlockNode): The aggregator node.
            transfer (Transfer): ...
            aggr_strategy (AggregatorStrategy): ...
            results (list[JobResult]): Results from children of ``node``.

        Returns:
            Aggregation results.
        """
        import pandas

        from flox.flock.states import AggrState, NodeState
        from flox.runtime import JobResult

        child_states: dict[NodeID, NodeState] = {}
        child_state_dicts = {}
        for result in results:
            idx: NodeID = result.node_idx
            child_states[idx] = result.node_state
            child_state_dicts[idx] = result.params

        node_state = AggrState(node.idx, children, None)  # FIXME
        avg_state_dict = aggr_strategy.aggregate_params(
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
        result = JobResult(node_state, node.idx, node.kind, avg_state_dict, history)
        return transfer.report(result)

    @property
    def __name__(self) -> str:
        return "AggregateJob"


class DebugAggregateJob(AggregableJob):
    @staticmethod
    def __call__(
        node: FlockNode,
        children: t.Iterable[FlockNode],
        transfer: BaseTransfer,
        aggr_strategy: AggregatorStrategy,
        results: list[Result],
    ) -> Result:
        """

        Args:
            node ():
            transfer ():
            aggr_strategy ():
            results ():

        Returns:

        """
        import datetime

        import numpy
        import pandas

        from flox.flock.states import AggrState
        from flox.runtime import JobResult

        result = next(iter(results))
        state_dict = result.params
        state_dict = {} if state_dict is None else state_dict
        node_state = AggrState(node.idx, children, None)
        history = {
            "node/idx": [node.idx],
            "node/kind": [node.kind.to_str()],
            "train/loss": [numpy.nan],
            "train/epoch": [numpy.nan],
            "train/batch_idx": [numpy.nan],
            "train/time": [datetime.datetime.now()],
            "mode": "debug",
        }
        history_df = pandas.DataFrame.from_dict(history)
        result = JobResult(node_state, node.idx, node.kind, state_dict, history_df)
        return transfer.report(result)
