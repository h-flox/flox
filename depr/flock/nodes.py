from enum import auto, Enum


class FlockNodes(Enum):
    leader = auto()
    aggregator = auto()
    worker = auto()


class LeaderNode:
    pass


class AggregatorNode:
    pass


class WorkerNode:
    pass
