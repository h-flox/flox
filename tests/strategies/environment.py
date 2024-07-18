from flight.federation.topologies.node import Node, NodeKind


def create_children(numWorkers: int, numAggr: int = 0) -> list[Node]:
    aggr = [Node(idx=i, kind=NodeKind.AGGR) for i in range(1, numAggr + 1)]
    workers = [
        Node(idx=i + numAggr, kind=NodeKind.WORKER) for i in range(1, numWorkers + 1)
    ]
    print(workers + aggr)
    print(aggr)
    return workers + aggr
