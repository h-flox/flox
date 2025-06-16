from __future__ import annotations

import datetime
import functools
import typing as t
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field

from .events import Context, CoordinatorEvents, fire_event_handler_by_type
from .jobs.aggr import AggregatorJobProto, AggrJobArgs, aggregator_job
from .jobs.protocols import Result
from .jobs.worker import WorkerJobArgs, worker_job
from .logging import init_logger
from .runtime import Runtime
from .state import WorkerState
from .system import Node
from .system.node import NodeKind

if t.TYPE_CHECKING:
    from .events import get_event_handlers_by_genre  # noqa: F401
    from .learning import TorchDataModule, TorchModule
    from .strategies.strategy import Strategy
    from .system.topology import Topology
    from .system.types import NodeID


def _fire_event_handler(
    obj: FederationWorkflow,
    event_type: CoordinatorEvents,
    context: dict[str, t.Any],
) -> None:
    """
    Helper alias function to fire an event handler by type.

    Args:
        obj (FederationWorkflow):
            The workflow object that contains the event handlers.
        event_type (CoordinatorEvents):
            The type of event to fire.
        context (dict[str, t.Any]):
            Contextual information to pass to the event handler.
    """
    fire_event_handler_by_type(obj, event_type, context)


def _resolve_ambiguous_node(topology: Topology, node: Node | None) -> Node:
    """
    Returns a node from a Topology in the case of ambiguity (i.e., the provided `Node`
    is `None`). In this case, the coordinator (root) of the topology is returned.
    Otherwise, the given `Node` is returned.

    Args:
        topology (Topology): The topology to resolve the node from.
        node (Node): The node to resolve.

    Returns:
        The resolved `Node` (the root if `node is None`; otherwise the `node` itself).

    Throws:
        - `TypeError`: If the `node` argument is neither a `Node` instance nor `None`.
    """
    match node:
        case None:
            return topology.coordinator
        case Node():
            return node
        case _:
            raise TypeError(
                f"`_resolve_ambiguous_node()` failed to resolve the arg `{node=}` to "
                f"a `Node` instance. Must either be a `Node` instance or `None` (only "
                f"if intended to be the Coordinator node)."
            )


def _set_parent_future(parent_fut: Future, child_fut: Future) -> t.Any:
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


def _all_futures_finished(
    job: AggregatorJobProto,
    args: AggrJobArgs,
    parent_fut: Future,
    child_futs: dict[NodeID, Future],
    runtime: Runtime,
    _: Future,
    # children: t.Iterable[Node],
    # node: Node,
    # aggr_strategy: AggrStrategy,
) -> None:
    """
    Callback function that is called when all child futures are finished.

    TODO

    Throws:
        - ValueError: Is thrown if `'__results'` is provided as a keyword argument.

    Returns:

    """
    # if _FUTURE_RESULTS_KEY in kwargs:
    #     raise ValueError(
    #         f"The key '{_FUTURE_RESULTS_KEY}' is reserved by Flight and cannot "
    #         f"be a keyword argument for `all_futures_finished`."
    #     )

    if all([fut.done() for _, fut in child_futs.items()]):
        args = AggrJobArgs(  # TODO: Fix this later.
            round_num=args.round_num,
            node=args.node,
            # children=args.children,
            child_results={idx: fut.result() for idx, fut in child_futs.items()},
            strategy=args.strategy,
            data_plane=runtime.data_plane,
        )
        try:
            fut = runtime.submit(job, args=args)
            cbk = functools.partial(_set_parent_future, parent_fut)
            fut.add_done_callback(cbk)
        except Exception as err:
            raise err


# TODO: Port this over into the `state` module.
@dataclass
class CoordinatorState:
    """
    The state of the coordinator node.
    """

    round: int = field(default=1)
    current_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    duration_time: datetime.timedelta = field(
        default_factory=lambda: datetime.datetime.now() - datetime.datetime.now()
    )
    start_time: datetime.datetime = field(default_factory=datetime.datetime.now)

    def update(self, incr_round: bool = False):
        self.current_time = datetime.datetime.now()
        self.duration_time = self.current_time - self.start_time

        if incr_round:
            self.round += 1


class FederationWorkflow:
    topology: Topology
    strategy: Strategy
    runtime: Runtime

    def __init__(
        self,
        topology: Topology,
        strategy: Strategy,
        module: TorchModule | None = None,
        dataset: TorchDataModule | None = None,
        num_rounds: int = 10,
    ):
        self.topology = topology
        self.strategy = strategy
        self.keep_going = False
        self.runtime = Runtime.simple_setup()

        self.num_rounds = num_rounds
        self.round_num = 0
        self.logger = init_logger()

        self.global_module = module
        self.dataset = dataset
        self._relevant_nodes = {}

    def start(self) -> None:
        """
        TODO
        """
        ctx: Context = {}
        state: CoordinatorState = CoordinatorState()
        self.keep_going = True
        self.round_num = 0
        _fire_event_handler(self, CoordinatorEvents.STARTED, ctx)

        while self.keep_going:
            _fire_event_handler(self, CoordinatorEvents.ROUND_STARTED, ctx)
            self.coordinator_round(state, ctx)
            _fire_event_handler(self, CoordinatorEvents.ROUND_COMPLETED, ctx)

        _fire_event_handler(self, CoordinatorEvents.COMPLETED, ctx)

    def coordinator_round(self, state: CoordinatorState, ctx):
        # WORKER SELECTION
        selected_workers = self.client_selection(ctx)
        relevant_nodes = get_relevant_nodes(self.topology, selected_workers)
        for node_idx in relevant_nodes:
            self.logger.info(f"[Round:{state.round}] - Launching job on {node_idx=}.")
            node = self.topology[node_idx]
            future = self.launch_jobs(node, None)
            future.result()

        state.update(incr_round=True)

        # TODO: Let's plan to implement a `Terminator` class/protocol that implements
        #       the logic to update this `keep_going` attribute. This can take in the
        #       `state` and `context` objects and update the `keep_going` attribute
        #       according to almost anything, number of rounds versus a total number of
        #       rounds for a federation, whether the test accuracy of the global model
        #       has converged by some threshold, etc.
        self.keep_going = state.round <= self.num_rounds

    def client_selection(self, ctx: Context = None) -> list[NodeID]:
        """

        Args:
            ctx (t.Any): Context to pass to the event handlers.

        Returns:

        """
        self.logger.info("Start of client selection phase.")
        _fire_event_handler(self, CoordinatorEvents.WORKER_SELECTION_STARTED, ctx)
        selected_workers = self.strategy.select_workers(self.topology)
        _fire_event_handler(self, CoordinatorEvents.WORKER_SELECTION_COMPLETED, ctx)
        self.logger.info("End of client selection phase.")
        return selected_workers

    def launch_jobs(
        self, node: Node | None = None, parent: Node | None = None
    ) -> Future[Result]:
        node = _resolve_ambiguous_node(self.topology, node)
        match node.kind:
            case NodeKind.COORDINATOR:
                return self.launch_coordinator_job()
            case NodeKind.AGGREGATOR:
                return self.launch_aggregator_job(node)
            case NodeKind.WORKER:
                if parent is None:
                    raise ValueError(
                        "Parent node must be provided (i.e., cannot be `None`) "
                        "for worker nodes."
                    )
                return self.launch_worker_job(node, parent)
            case _:
                raise ValueError("Invalid node kind.")

    def launch_coordinator_job(self) -> Future[Result]:
        node = self.topology.coordinator
        return self.launch_aggregator_job(node)

    def launch_aggregator_job(self, node: Node) -> Future[Result]:
        parent_future: Future = Future()
        children_futures: list[Future] = [
            self.launch_jobs(node=child, parent=node)
            for child in self._relevant_nodes[node.idx]
        ]

        _children: list[Node] = self.topology.children(node)  # noqa: F841
        args = AggrJobArgs(
            round_num=self.round_num,
            node=node,
            child_results=dict(),  # This is updated in the callback (defined below).
            strategy=self.strategy,
            data_plane=self.runtime.data_plane,
        )
        callback = functools.partial(
            _all_futures_finished,
            aggregator_job,
            args,
            parent_future,
            children_futures,
            self.runtime,
        )
        for fut in children_futures:
            fut.add_done_callback(callback)

        return parent_future

    def launch_worker_job(self, node: Node, parent: Node) -> Future[Result]:
        state = WorkerState()  # noqa: F841
        args = WorkerJobArgs(
            strategy=self.strategy,
            model=self.global_module,
            data=self.dataset,
            params=self.global_module.get_params(),
            node=node,
            parent=parent,
            # ...
        )
        # args = self.runtime.transfer(args)
        return self.runtime.submit(worker_job, args=args)


def get_relevant_nodes(
    topology: Topology,
    selected_workers: list[NodeID],
) -> dict[NodeID, list[Node | NodeID]]:
    """
    Given a set of selected workers, this function returns _all_ the relevant nodes
    (that are _not_ the `Coordinator`) to the execution of a federation.

    Args:
        topology (Topology): ...
        selected_workers (list[NodeID]): ...

    Returns:
        The relevant nodes to the federation, not including the `Coordinator` node.

    Examples:
        ```mermaid
        flowchart TB
            c[Coordinator: 0]
            w1((Worker: 1))
            w2((Worker: 2))
            w3((Worker: 3))

            c-->w1
            c-->w2
            c-->w3

            style w1 stroke-width:3px,stroke-dasharray: 5 5
            style w3 stroke-width:3px,stroke-dasharray: 5 5
        ```

        >>> topo = ... # above topology
        >>> get_event_handlers_by_genre(topo, [1, 3])
        {0: [1, 3]}

        ```mermaid
        flowchart TB
            c[Coordinator]
            a{{Aggregator}}
            w1((Worker-1))
            w2((Worker-2))
            w3((Worker-3))

            c-->a
            c-->w3
            a-->w1
            a-->w2

            style w1 stroke-width:3px,stroke-dasharray: 5 5
            style w3 stroke-width:3px,stroke-dasharray: 5 5
            style a stroke-width:3px,stroke-dasharray: 5 5
        ```
        >>> topo = ... # above topology
        >>> get_event_handlers_by_genre(topo, [1, 3])
        {"Coordinator": ["Worker-1", "Worker-3", "Aggregator"]}
    """
    middle_aggrs: set[NodeID] = set()
    for worker in selected_workers:
        parent = topology.parent(worker)
        while parent != topology.coordinator:
            middle_aggrs.add(parent.idx)
            parent = topology.parent(parent)

    selected_workers = set(selected_workers)
    coord_children = {child.idx for child in topology.children(topology.coordinator)}

    if len(middle_aggrs) > 0:
        selected_coord_children = coord_children.intersection(middle_aggrs)
    else:
        selected_coord_children = coord_children.intersection(selected_workers)

    selected_children: dict[NodeID, list[NodeID]] = {
        topology.coordinator.idx: list(selected_coord_children)
    }
    for aggr in middle_aggrs:
        relevant_aggrs_and_workers = middle_aggrs.union(selected_workers)
        aggr_children = {child.idx for child in topology.children(aggr)}
        aggr_children = aggr_children.intersection(relevant_aggrs_and_workers)
        selected_children[aggr] = list(aggr_children)

    return selected_children
