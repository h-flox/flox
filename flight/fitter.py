import typing as t

from flight.events import CoordinatorEvents


def simple_federated_fit(
    topology: t.Any,
    rounds: int,
):
    """Skeletal framework for how FL will be performed in Flight with events."""
    CoordinatorEvents.STARTED

    curr_round: int = 0
    while True:
        CoordinatorEvents.ROUND_STARTED

        federated_round(topology)

        CoordinatorEvents.ROUND_COMPLETED

        curr_round += 1
        cond = curr_round >= rounds
        if cond:
            break

    CoordinatorEvents.COMPLETED


def federated_round(topology):
    CoordinatorEvents.WORKER_SELECTION_STARTED
    selected_workers = worker_selection([1, 2, 3])
    CoordinatorEvents.WORKER_SELECTION_COMPLETED

    relevant_nodes = get_relevant_nodes(topology, selected_workers)

    return  # TODO: Edit later.
    for node in relevant_nodes:
        if node.kind == "aggregator":
            pass
        elif node.kind == "worker":
            pass
        else:
            pass


def worker_selection(lst: list[t.Any]) -> list[t.Any]:
    return lst


def get_relevant_nodes(topology: t.Any, selected_workers: list[t.Any]):
    return selected_workers
