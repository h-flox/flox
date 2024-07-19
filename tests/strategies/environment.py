from flight.federation.topologies.node import Node, NodeKind


def create_children(num_workers: int, num_aggrs: int = 0) -> list[Node]:
    """Creates a fabricated list of children used for coordinator/selecting workers test cases.

    Args:
        num_workers (int): Number of workers to be added.
        num_aggrs (int, optional): Number of aggregators to be added. Defaults to 0.

    Returns:
        list[Node]: A list of the created children.
    """
    aggr = [Node(idx=i, kind=NodeKind.AGGR) for i in range(1, num_aggrs + 1)]
    workers = [
        Node(idx=i + num_aggrs, kind=NodeKind.WORKER) for i in range(1, num_workers + 1)
    ]
    return workers + aggr
