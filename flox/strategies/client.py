from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flox.federation.topologies import Node, NodeState


class ClientStrategy(t.Protocol):
    # def get_node_statuses(self):
    #     pass

    def select_worker_nodes(
        self, state: NodeState, children: t.Iterable[Node], seed: int | None
    ) -> t.Iterable[Node]:
        """Callback that is responsible with selecting a subset of worker nodes to do local training.

        Args:
            state (NodeState): The state of the client node.
            children (t.Iterable[Node]): The worker nodes in the topology.
            seed (int): The seed to use for reproducibility.

        Returns:
            Selected worker nodes.
        """
        pass

    # def before_share_params(self):
    #     pass
