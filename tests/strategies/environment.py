from flight.federation.topologies.node import Node, NodeKind


def create_children(numWorkers: int, numAggr: int = 0) -> list[Node]:
    """Creates a fabricated list of children used for coordinator/selecting workers test cases.

    Args:
        numWorkers (int): Number of workers to be added.
        numAggr (int, optional): Number of aggregators to be added. Defaults to 0.

    Returns:
        list[Node]: A list of the created children.
    """
    aggr = [Node(idx=i, kind=NodeKind.AGGR) for i in range(1, numAggr + 1)]
    workers = [
        Node(idx=i + numAggr, kind=NodeKind.WORKER) for i in range(1, numWorkers + 1)
    ]
    return workers + aggr
