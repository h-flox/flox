import pytest

from flight.system.node import *


def test_node_kind_init():
    coord = NodeKind("coordinator")
    assert isinstance(coord, NodeKind)
    assert coord is NodeKind.COORDINATOR

    aggr = NodeKind("aggregator")
    assert isinstance(aggr, NodeKind)
    assert aggr is NodeKind.AGGREGATOR

    wrkr = NodeKind("worker")
    assert isinstance(wrkr, NodeKind)
    assert wrkr is NodeKind.WORKER

    with pytest.raises(ValueError):
        NodeKind("other")


def test_node_coordinator_init():
    node = Node(1, "coordinator")
    assert isinstance(node, Node)
    assert node.kind is NodeKind.COORDINATOR
    assert node.get("key", default=10) == 10

    with pytest.raises(ValueError):
        Node(1, "other")


def test_node_aggregator_init():
    node = Node(1, "aggregator")
    assert isinstance(node, Node)
    assert node.kind is NodeKind.AGGREGATOR
    assert node.get("key", default=10) == 10

    with pytest.raises(ValueError):
        Node(1, "other")


def test_node_worker_init():
    node = Node(1, "worker")
    assert isinstance(node, Node)
    assert node.kind is NodeKind.WORKER
    assert node.get("key", default=10) == 10

    with pytest.raises(ValueError):
        Node(1, "other")
