from enum import Enum, auto


class FlockNodeStatus(Enum):
    UNAVAILABLE = auto()
    AVAILABLE = auto()
    RUNNING = auto()
