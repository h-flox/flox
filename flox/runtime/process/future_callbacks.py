import functools
from concurrent.futures import Future

from flox.flock import FlockNode
from flox.runtime.jobs import aggregation_job
from flox.runtime.runtime import Runtime
from flox.runtime.utils import set_parent_future
from flox.strategies import Strategy


def all_child_futures_finished_cbk(
    parent_future: Future,
    children_futures: list[Future],
    node: FlockNode,
    runtime: Runtime,
    strategy: Strategy,
    _: Future,
):
    if all([child_future.done() for child_future in children_futures]):
        # TODO: We need to add error-handling for cases when the
        #       `TaskExecutionFailed` error from Globus-Compute is thrown.
        children_results = [child_future.result() for child_future in children_futures]
        future = runtime.submit(
            aggregation_job,
            node,
            strategy=strategy,
            results=children_results,
        )
        aggr_done_callback = functools.partial(set_parent_future, parent_future)
        future.add_done_callback(aggr_done_callback)
