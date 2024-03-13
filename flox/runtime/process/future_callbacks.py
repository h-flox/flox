from __future__ import annotations

import functools
import typing

from flox.runtime.utils import set_parent_future

if typing.TYPE_CHECKING:
    from concurrent.futures import Future

    from flox.flock import FlockNode
    from flox.jobs import Job
    from flox.runtime.runtime import Runtime
    from flox.strategies import AggregatorStrategy


def all_child_futures_finished_cbk(
    job: Job,
    parent_future: Future,
    children_futures: typing.Iterable[Future],
    node: FlockNode,
    runtime: Runtime,
    aggr_strategy: AggregatorStrategy,
    _: Future,
):
    if all([child_future.done() for child_future in children_futures]):
        # TODO: We need to add error-handling for cases when the
        #       `TaskExecutionFailed` error from Globus-Compute is thrown.
        children_results = [child_future.result() for child_future in children_futures]
        future = runtime.submit(
            job,
            node=node,
            aggr_strategy=aggr_strategy,
            results=children_results,
        )
        aggr_done_callback = functools.partial(set_parent_future, parent_future)
        future.add_done_callback(aggr_done_callback)
