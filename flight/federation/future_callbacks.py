"""
Defines callbacks for processing a set of futures.
"""
from __future__ import annotations

import collections as c
import functools
import typing as t
from concurrent.futures._base import InvalidStateError  # noqa

from flight.federation.jobs.types import AggrJobArgs

if t.TYPE_CHECKING:
    from concurrent.futures import Future

    from flight.engine import Engine
    from flight.federation.jobs.types import AggrJob

_FUTURE_RESULTS_KEY: t.Final[str] = "__results"

CallbackArgs = c.namedtuple("CallbackArgs", ["parent_fut", "child_fut", "engine"])


def set_parent_future(parent_fut: Future, child_fut: Future) -> t.Any:
    if not child_fut.done():
        raise ValueError(
            "set_parent_future(): Arg `child_fut` must be done "
            "(i.e., `child_fut.done() == True`)."
        )
    elif child_fut.exception():
        parent_fut.set_exception(child_fut.exception())
    else:
        result = child_fut.result()
        try:
            parent_fut.set_result(result)
        except InvalidStateError:
            pass
        return result


def all_futures_finished(
    job: AggrJob,
    args: AggrJobArgs,
    parent_fut: Future,
    child_futs: t.Iterable[Future],
    engine: Engine,
    _: Future,
    # children: t.Iterable[Node],
    # node: Node,
    # aggr_strategy: AggrStrategy,
) -> None:
    """

    Args:
        job (AggrJob):
        args (AggrJobArgs):
        parent_fut (Future):
        child_futs (typing.Iterable[Future]):
        engine:
        _:

    Throws:
        - ValueError: Is thrown if `'__results'` is provided as a keyword argument.

    Returns:

    """
    # if _FUTURE_RESULTS_KEY in kwargs:
    #     raise ValueError(
    #         f"The key '{_FUTURE_RESULTS_KEY}' is reserved by Flight and cannot "
    #         f"be a keyword argument for `all_futures_finished`."
    #     )

    if all([fut.done() for fut in child_futs]):
        args = AggrJobArgs(  # TODO: Fix this later.
            round_num=args.round_num,
            node=args.node,
            children=args.children,
            child_results=[fut.result() for fut in child_futs],
            aggr_strategy=args.aggr_strategy,
            transfer=args.transfer,
        )
        try:
            fut = engine.submit(job, args=args)
            cbk = functools.partial(set_parent_future, parent_fut)
            fut.add_done_callback(cbk)
        except Exception as err:
            raise err
