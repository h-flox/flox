from __future__ import annotations

import functools
import typing as t

from flox.runtime.utils import set_parent_future

if t.TYPE_CHECKING:
    from concurrent.futures import Future

    from flox.flock import FlockNode
    from flox.jobs import AggregableJob
    from flox.runtime.runtime import Runtime
    from flox.strategies import AggregatorStrategy


def all_child_futures_finished_cbk(
    job: AggregableJob,
    parent_future: Future,
    children: t.Iterable[FlockNode],
    selected_children_futures: t.Iterable[Future],
    # global_model: FloxModule,
    node: FlockNode,
    runtime: Runtime,
    aggr_strategy: AggregatorStrategy,
    _: Future,
) -> None:
    """
    This partial function (`subtree_done_cbk`) will perform the aggregation only
    when all futures in `children_futures` has completed. This partial function
    will be added as a callback which is run after the completion of each child
    future. But, it will only perform aggregation once since only the last future
    to be completed will activate the conditional.
    """
    if all([child_future.done() for child_future in selected_children_futures]):
        # TODO: We need to add error-handling for cases when the
        #       `TaskExecutionFailed` error from Globus-Compute is thrown.
        children_results = [fut.result() for fut in selected_children_futures]
        future = runtime.submit(
            job,
            node=node,
            children=children,
            aggr_strategy=aggr_strategy,
            results=children_results,
        )
        callback = functools.partial(set_parent_future, parent_future)
        future.add_done_callback(callback)
