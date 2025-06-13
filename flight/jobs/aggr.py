from __future__ import annotations

import typing as t
from dataclasses import dataclass

if t.TYPE_CHECKING:
    from flight.strategies.strategy import Strategy
    from flight.system.topology import NodeID

    from ..learning.parameters import Params
    from ..system.node import Node
    from .protocols import Result


@dataclass
class AggrJobArgs:
    """
    The arguments for the aggregator job.
    """

    node: Node
    child_results: list[Result]
    round_num: int
    handlers: list[t.Any]
    strategy: Strategy


class AggregatorJobProto(t.Protocol):
    @staticmethod
    def __call__(args: AggrJobArgs) -> Result:
        """
        This method is called when the AGGREGATOR job is launched.
        """


def aggregator_job(args: AggrJobArgs) -> Result:
    from flight.learning.module import TorchModule
    from flight.state import AggregatorState, WorkerState

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

    for result in args.child_results:
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
        module=aggr_module,
        params=aggr_params,
        extra={
            "child_states": child_states,
            "child_modules": child_modules,
            "round_num": args.round_num,
        },
    )
