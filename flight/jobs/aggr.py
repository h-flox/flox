from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

if t.TYPE_CHECKING:
    from flight.strategies.strategy import Strategy
    from flight.system.topology import NodeID

    from ..commons import Record
    from ..learning.parameters import Params
    from ..runtime import DataPlane
    from ..system.node import Node
    from .protocols import Result


@dataclass
class AggrJobArgs:
    """
    The arguments for the aggregator job.
    """

    node: Node
    child_results: dict[NodeID, Result]
    round_num: int
    strategy: Strategy
    data_plane: DataPlane | None = field(default=None, repr=False)


class AggregatorJobProto(t.Protocol):
    @staticmethod
    def __call__(args: AggrJobArgs) -> Result:
        """
        This method is called when the AGGREGATOR job is launched.
        """


def aggregator_job(args: AggrJobArgs) -> Result:
    """

    Args:
        args:

    Returns:
        ...

    Throws:
        - `TypeError`:
            - If the `node` argument is not an instance of `Node`.
            - If the `state` of any child result is not an instance of
              `AggregatorState` or `WorkerState`.
    """
    from flight.jobs.protocols import Result
    from flight.learning.module import TorchModule
    from flight.state import AggregatorState, WorkerState
    from flight.system import Node

    if not isinstance(args.node, Node):
        raise TypeError(
            "Node must be an instance of `Node`. Aggregation requires a `Topology` "
            "to be performed in Flight. The `node` argument can only be `None` in "
            "the case where you run the worker job for local testing outside running "
            "a federation workflow."
        )

    child_states: dict[NodeID, AggregatorState | WorkerState] = {}
    child_params: dict[NodeID, Params] = {}
    child_modules: dict[NodeID, TorchModule] = {}
    child_records: list[Record] = []

    for _child_idx, result in args.child_results.items():
        if isinstance(result.state, (AggregatorState, WorkerState)):
            child_states[result.node.idx] = result.state
        else:
            raise TypeError(
                f"Child state {result.node.idx} must be an instance of "
                "`AggregatorState` or `WorkerState`, got {type(result.state)}."
            )

        if isinstance(result.module, TorchModule):
            child_modules[result.node.idx] = result.module
        else:
            raise TypeError(
                f"Child module {result.node.idx} must be an instance of "
                f"`TorchModule`, got {type(result.module)}."
            )

        if result.params is None:
            child_params[result.node.idx] = result.module.get_params()
        else:
            child_params[result.node.idx] = result.params

        child_records.extend(result.records)

        # if result.usable():
        #     child_states[result.node.idx] = result.state
        #     child_params[result.node.idx] = result.params
        #     child_modules[result.node.idx] = result.module
        #
        #     if not isinstance(result.state, (AggregatorState, WorkerState)):
        #         raise TypeError(
        #             f"Child state {result.node.idx} must be an instance of "
        #             "`AggregatorState` or `WorkerState`, got {type(result.state)}."
        #         )
        # else:
        #     raise ValueError(
        #         f"Child result {result.node.idx} is not usable (see "
        #         f"`Result.usable()`) . Ensure that the child job completed "
        #         f"successfully."
        #     )

    aggr_state = AggregatorState()
    aggr_params = args.strategy.aggregate(child_params)
    aggr_module = next(iter(child_modules.values()))

    return Result(
        node=args.node,
        state=aggr_state,
        records=child_records,
        module=aggr_module,
        round_num=args.round_num,
        params=aggr_params,
        extra={
            "child_states": child_states,
            "child_modules": child_modules,
            "round_num": args.round_num,
        },
    )
