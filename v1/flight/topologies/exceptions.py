class TopologyException(Exception):
    """
    A general exception that is thrown when an exception related to the topology occurs.
    """


class IllegalTopologyError(Exception):
    """An Exception that is raised when a topology has an illegal structure."""

    def __init__(self, message: str | None = None, *args):
        if message is None:
            message = (
                "The structure of the Topology is not legal. Please refer to the docs "
                "for more information on the topological requirements."
            )
        super().__init__(message, *args)


class NodeNotFoundError(Exception):
    """An Exception that is raised when a node is not in a `Topology`."""

    def __init__(self, message: str | None = None, *args):
        if message is None:
            message = "The given `NodeID` is not part of this topology."
        super().__init__(message, *args)
