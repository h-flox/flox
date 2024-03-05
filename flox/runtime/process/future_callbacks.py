from __future__ import annotations

import functools
import typing

from flox.jobs import AggregateJob
from flox.runtime.utils import set_parent_future

if typing.TYPE_CHECKING:
    from concurrent.futures import Future

    from flox.flock import FlockNode
    from flox.runtime.runtime import Runtime
    from flox.strategies_depr import Strategy


def all_child_futures_finished_cbk(
    parent_future: Future,
    children_futures: typing.Iterable[Future],
    node: FlockNode,
    runtime: Runtime,
    strategy: Strategy,
    _: Future,
):
    if all([child_future.done() for child_future in children_futures]):
        # TODO: We need to add error-handling for cases when the
        #       `TaskExecutionFailed` error from Globus-Compute is thrown.
        children_results = [child_future.result() for child_future in children_futures]
        job = AggregateJob()
        future = runtime.submit(
            job,
            node,
            strategy=strategy,
            results=children_results,
        )
        aggr_done_callback = functools.partial(set_parent_future, parent_future)
        future.add_done_callback(aggr_done_callback)
