import pytest

import networkx as nx
import typing as t

from flight.events import *
from flight.fitter import simple_federated_fit
from flight.strategy import Strategy, DefaultStrategy
from flight.system.topology import Topology


class TestStrategy(DefaultStrategy):
    # TODO: We need to fix the `|` operator for events.
    @on(
        CoordinatorEvents.STARTED # |
        # CoordinatorEvents.COMPLETED |
        # CoordinatorEvents.ROUND_STARTED |
        # CoordinatorEvents.ROUND_COMPLETED |
        # CoordinatorEvents.WORKER_SELECTION_STARTED |
        # CoordinatorEvents.WORKER_SELECTION_COMPLETED
    )
    def count(self, context):
        print(context)
        context["count"] += 1


@pytest.fixture
def topology() -> Topology:
    graph = nx.star_graph(11)
    graph = graph.to_directed()
    graph.remove_edges_from(filter(lambda edge: edge[1] == 0, list(graph.edges())))

    for node, node_data in graph.nodes(data=True):
        node_data["kind"] = "coordinator" if node == 0 else "worker"
    
    return Topology.from_networkx(graph)
    

def test_federated_fit_without_failure(topology):
    try:
        simple_federated_fit(topology, DefaultStrategy(), 10)
        assert True
    except BaseException as err:
        pytest.fail(f"Unexpected error: {err}")


def test_federated_fit_event_hooks(topology):
    strategy = TestStrategy()
    handlers = strategy.get_event_handlers(CoordinatorEvents.STARTED)
    print(handlers)
    context = simple_federated_fit(topology, strategy=strategy, rounds=1)
    print(context)
    assert context["count"] == 6
