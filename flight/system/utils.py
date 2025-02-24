from .node import Node
from .topology import Topology


def flat_topology(n: int, /, **kwargs) -> Topology:
    """Creates a flat `Topology` where Workers are directly connected to
    the Coordinator.

    Args:
        n (int): Number of worker nodes.

    Returns:
        A flat topology with `n + 1` total nodes.
    """
    edges = []
    nodes = [Node(0, "coordinator", **kwargs)]

    for i in range(1, n + 1):
        nodes.append(Node(i, "worker", **kwargs))
        edges.append((0, i))

    return Topology(nodes, edges)
