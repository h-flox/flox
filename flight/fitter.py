from __future__ import annotations

import typing as t

from .events import CoordinatorEvents
from .runtime import Runtime

if t.TYPE_CHECKING:
    from .strategy import Strategy
    from .system.topology import Topology


def simple_federated_fit(
    topology: Topology,
    strategy: Strategy,
    rounds: int,
):
    context: dict[str, t.Any] = {}
    runtime = Runtime()

    """Skeletal framework for how FL will be performed in Flight with events."""
    strategy.fire_event_handler(CoordinatorEvents.STARTED, context)

    curr_round: int = 0
    while True:
        strategy.fire_event_handler(CoordinatorEvents.ROUND_STARTED, context)

        federated_round(runtime, topology, strategy, context)

        strategy.fire_event_handler(CoordinatorEvents.ROUND_COMPLETED, context)

        curr_round += 1
        cond = curr_round >= rounds
        if cond:
            break

    strategy.fire_event_handler(CoordinatorEvents.COMPLETED, context)
    return context


def federated_round(
    runtime: Runtime,
    topology: Topology,
    strategy: Strategy,
    context: dict[str, t.Any],
):
    strategy.fire_event_handler(CoordinatorEvents.WORKER_SELECTION_STARTED, context)
    selected_workers = worker_selection([1, 2, 3])
    strategy.fire_event_handler(CoordinatorEvents.WORKER_SELECTION_COMPLETED, context)

    relevant_nodes = get_relevant_nodes(topology, selected_workers)

    return  # TODO: Edit later.
    for node in relevant_nodes:
        if node.kind == "aggregator":
            runtime.submit(...)

        elif node.kind == "worker":
            runtime.submit(...)

        else:
            raise ValueError


def worker_selection(lst: list[t.Any]) -> list[t.Any]:
    return lst


def get_relevant_nodes(topology: t.Any, selected_workers: list[t.Any]):
    return selected_workers
