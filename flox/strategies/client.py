from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flox.flock import NodeID


class ClientStrategy(t.Protocol):
    # def get_node_statuses(self):
    #     pass

    def select_worker_nodes(self, state, children, seed) -> t.Iterable[NodeID]:
        pass

    # def before_share_params(self):
    #     pass
