from __future__ import annotations

import datetime
import typing as t
from dataclasses import dataclass, field

from .events import (
    AggregatorEvents,
    CoordinatorEvents,
    WorkerEvents,
    fire_event_handler_by_type,
    get_event_handlers_by_genre,
)
from .jobs.worker import worker_job
from .logging import init_logger
from .runtime import Runtime
from .system.topology import NodeKind

if t.TYPE_CHECKING:
    from concurrent.futures import Future

    from .strategy import Strategy
    from .system.topology import Topology
    from .system.types import NodeID


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
        num_rounds: int = 10,
    ):
        self.topology = topology
        self.strategy = strategy
        self.keep_going = False
        self.runtime = Runtime.simple_setup()

        self.num_rounds = num_rounds
        self.logger = init_logger()

    def start(self) -> None:
        """
        TODO
        """
        context: dict[str, t.Any] = {}
        state: CoordinatorState = CoordinatorState()
        self.keep_going = True
        self._fire_event_handler(CoordinatorEvents.STARTED, context)

        while self.keep_going:
            self._fire_event_handler(CoordinatorEvents.ROUND_STARTED, context)
            self.coordinator_round(state, context)
            self._fire_event_handler(CoordinatorEvents.ROUND_COMPLETED, context)

        self._fire_event_handler(CoordinatorEvents.COMPLETED, context)

    def coordinator_round(self, state: CoordinatorState, context):
        # WORKER SELECTION
        selected_workers = self.client_selection(context)
        relevant_nodes = get_relevant_nodes(self.topology, selected_workers)
        for node in relevant_nodes:
            self.logger.info(f"[Round:{state.round}] - Launching job on {node=}.")
            self.launch_node_jobs(node, None)

        state.update(incr_round=True)

        # TODO: Let's plan to implement a `Terminator` class/protocol that implements
        #       the logic to update this `keep_going` attribute. This can take in the
        #       `state` and `context` objects and update the `keep_going` attribute
        #       according to almost anything, number of rounds versus a total number of
        #       rounds for a federation, whether the test accuracy of the global model
        #       has converged by some threshold, etc.
        self.keep_going = state.round <= self.num_rounds

    def client_selection(self, context: t.Any = None) -> list[NodeID]:
        self.logger.info("Start of client selection phase.")
        self._fire_event_handler(CoordinatorEvents.WORKER_SELECTION_STARTED, context)
        selected_workers = self.strategy.select_workers(self.topology)
        self._fire_event_handler(CoordinatorEvents.WORKER_SELECTION_COMPLETED, context)
        self.logger.info("End of client selection phase.")
        return selected_workers

    def launch_node_jobs(
        self,
        node_id: NodeID | None = None,
        parent: NodeID | None = None,
    ) -> Future:
        if node_id is None:
            node = self.topology.coordinator
        else:
            node = self.topology[node_id]

        if node.kind is NodeKind.AGGREGATOR:

            def aggr_job(x):
                return None  # TODO

            event_handlers = get_event_handlers_by_genre(  # noqa
                self.strategy,
                AggregatorEvents,
            )
            return self.runtime.submit(aggr_job, ...)

        elif node.kind is NodeKind.WORKER:
            worker_event_handlers = get_event_handlers_by_genre(  # noqa
                self.strategy,
                WorkerEvents,
            )
            return self.runtime.submit(worker_job, ...)

        else:
            raise ValueError("Invalid node kind.")

    def _fire_event_handler(
        self,
        event: CoordinatorEvents,
        context: dict[str, t.Any],
    ) -> None:
        """
        Shorthand function for
        [`Strategy.fire_event_handler`][flight.strategy.Strategy.fire_event_handler]
        that uses the `Workflow`'s `Strategy` implementation.
        """
        fire_event_handler_by_type(self, event, context)


def get_relevant_nodes(
    topology: Topology,
    selected_workers: list[NodeID],
) -> list[NodeID]:
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
        [1, 3]

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
        ["Worker-1", "Worker-3", "Aggregator"]
    """
    return selected_workers
