import copy
import functools
import typing as t

from collections import defaultdict
from concurrent.futures import Future

from v1.flight import Topology
from v1.flight.engine import Engine
from v1.flight.engine.controllers.local import LocalController
from v1.flight.engine.transporters import InMemoryTransporter
from v1.flight.federation.future_callbacks import all_futures_finished
from v1.flight.learning import AbstractModule, AbstractDataModule
from v1.flight.topologies import Node, NodeKind
from v1.flight.federation.types import Result
from v1.flight.federation_v2.jobs.aggregation import aggregation_job
from v1.flight.federation_v2.jobs.ignite import training_job
from v1.flight.federation_v2.events import CoordEvent, WorkerEvent, IgniteEvent

from v1.flight.strategies_v2.sync.base import Strategy

ROUNDS: int = 10
AGGR_JOB = aggregation_job
WORKER_JOB = training_job

## TODO: Add a `logger` parameter to the Event objects.


# class Strategy:
#     def __init__(self):
#         pass


class SyncFederationV2:
    def __init__(
        self,
        topology: Topology,
        module: AbstractModule,
        data: AbstractDataModule,
        strategy: Strategy,
    ):
        self.topology = topology
        self.global_module = module
        self.data = data
        self.strategy = strategy

        self.engine = Engine(
            LocalController("process"),
            InMemoryTransporter(),
        )

        self.relevant_nodes: dict[Node, list[Node]] = defaultdict(list)
        self.round_num: int = 0

    def start(self):
        topology = self.topology
        strategy = self.strategy
        context = locals()

        for rnd in range(ROUNDS):
            strategy.fire_events_by_type(CoordEvent.ROUND_START, context)

            # Coordinator-side worker selection
            # strategy.fire_events_by_type(EventType.FOO, context)
            # BEFORE WORKER SELECTION
            selected_workers = self.strategy.selection_policy(...)
            relevant_nodes = find_relevant_nodes(self.topology, selected_workers)
            self.relevant_nodes.update(relevant_nodes)
            # AFTER WORKER SELECTION
            # strategy.fire_events_by_type(EventType.BAR, context)

            self.run_node_jobs()

            # Intermediate aggregator-side aggregation
            # for aggregator in aggregators:
            #     strategy.fire_events_by_type(AggrEvent.BEFORE_AGGR, context)
            #     aggregated_model = strategy.aggregation_policy(workers)
            #     strategy.fire_events_by_type(AggrEvent.AFTER_AGGR, context)

            # Coordinator-side aggregation
            # strategy.fire_events_by_type(CoordEvent.BEFORE_AGGR, context)
            # aggregated_model = strategy.aggregation_policy(workers)
            # strategy.fire_events_by_type(CoordEvent.AFTER_AGGR, context)

            """
            # TESTING: refer to IGNITE for ideas of which events go here.
            # Coordinator-side global model evaluation/testing.
            strategy.fire_events_by_type(CoordEvent.BEFORE_TEST_DATA_LOAD, context)
            test_data: Dataset = ...
            strategy.fire_events_by_type(CoordEvent.AFTER_TEST_DATA_LOAD, context)
            test_loader: DataLoader = ...
    
            tester: engine.Engine = engine.create_supervised_evaluator(...)
            for event, handler in strategy.get_event_handlers_by_type(IgniteEvent):
                tester.add_event_handler(event, handler, context)
            tester.run(...)
            """

            strategy.fire_events_by_type(CoordEvent.ROUND_END, context)

    def run_node_jobs(self, node: Node | None = None, parent: Node | None = None):
        node = _resolve_ambiguous_node(self.topology, node)
        match node.nind:
            case NodeKind.COORD | NodeKind.AGGR:
                # EVENT.BEFORE_COORD_JOB
                return self.run_coordinator_job()

            case NodeKind.AGGR:
                # EVENT.BEFORE_AGGR_JOB
                return self.run_aggregator_job(node)

            case NodeKind.WORKER:
                if parent is None:
                    raise ValueError(
                        "Parent node must be provided (i.e., cannot be `None`) "
                        "for worker nodes."
                    )

                # EVENT.BEFORE_WORKER_JOB
                return self.run_worker_job(node, parent)

            case _:
                raise ValueError("Invalid node kind.")

    def run_coordinator_job(self) -> Future[Result]:
        node = self.topology.coordinator
        return self.run_aggregator_job(node)

    def run_aggregator_job(self, node: Node) -> Future[Result]:
        parent_future: Future = Future()
        children_futures: list[Future] = [
            self.run_node_jobs(node=child, parent=node)
            for child in self.relevant_nodes[node]
        ]

        children = list(self.topology.children(node))
        args = dict(
            round_num=self.round_num,
            node=node,
            children=children,
            child_results=[],  # This is updated in the callback (defined below).
            strategy=self.strategy,
            transmitter=self.engine.transmitter,
        )
        cbk = functools.partial(all_futures_finished, args)
        for fut in children_futures:
            fut.add_done_callback(cbk)

        return parent_future

    def run_worker_job(self, node: Node, parent: Node) -> Future[Result]:
        args = dict(
            round_num=self.round_num,
            node=node,
            parent=parent,
            model=copy.deepcopy(self.global_module),
            data=self.data,
            worker_handlers=self.strategy.get_event_handlers_by_type(WorkerEvent),
            learning_handlers=self.strategy.get_event_handlers_by_type(IgniteEvent),
        )
        args = self.engine.transfer(args)
        return self.engine.submit(WORKER_JOB, args=args)


def find_relevant_nodes(
    topo: Topology, selected_workers: list[Node]
) -> dict[Node, list[Node]]:
    """
    Given a list of worker nodes selected to train (in a given round) on a topology,
    this function returns a map where each key is a node and its value is the list
    of "relevant" nodes for this round (based on `selected_workers`).

    A "relevant" node is a node that is either a selected worker node or an intermediate
    aggregator node that lies on the path between a selected worker node (or nodes) and
    the central coordinator.

    Args:
        topo (Topology): The topology to operate on.
        selected_workers (list[Node]): The list of worker nodes selected for this round.

    Returns:
        A dictionary where each key is a node and its value is the list of
        relevant nodes according to `selected_workers`.
    """
    # ...
    middle_aggrs = set()
    for worker in selected_workers:
        parent = topo.parent(worker)
        while parent != topo.coordinator:
            middle_aggrs.add(parent)
            parent = topo.parent(parent)

    selected_workers_set = set(selected_workers)

    # ...
    coord_children = set(topo.children(topo.coordinator))
    if not middle_aggrs:
        selected_workers = list(coord_children.intersection(selected_workers_set))
        return {topo.coordinator: selected_workers}

    # ...
    relevant_nodes = {topo.coordinator: coord_children.intersection(middle_aggrs)}
    relevant_aggrs_and_workers = middle_aggrs.union(selected_workers_set)
    for aggr in middle_aggrs:
        child_set = set(topo.children(aggr))
        relevant_children = child_set.intersection(relevant_aggrs_and_workers)
        relevant_nodes[aggr] = list(relevant_children)

    return relevant_nodes


@t.overload
def _resolve_ambiguous_node(topology: Topology, node: Node) -> Node:
    ...


@t.overload
def _resolve_ambiguous_node(topology: Topology, node: None) -> Node:
    ...


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
