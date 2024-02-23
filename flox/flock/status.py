from enum import auto, Enum


class FlockNodeStatus(Enum):
    UNAVAILABLE = auto()
    AVAILABLE = auto()
    RUNNING = auto()
