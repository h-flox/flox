from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from numpy.random import Generator

    from flight.federation.topologies.node import Node, NodeState


@t.runtime_checkable
class CoordStrategy(t.Protocol):
    """
    Protocol for all coordinator strategies, including those defined in Flight and
    those defined by users.
    """

    def select_workers(
        self,
        state: NodeState,
        workers: t.Iterable[Node],
        rng: Generator,
    ) -> t.Sequence[Node]:
        """
        Callback that is responsible for selecting a subset of worker nodes
        to do local training.

        Args:
            state (NodeState): The state of the current coordinator node.
            workers (t.Iterable[Node]): The worker nodes in the topology.
            rng (Generator): The rng used for reproducibility.

        Returns:
            The selected worker nodes.
        """
