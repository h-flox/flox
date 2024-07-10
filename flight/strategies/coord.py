from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from numpy.random import Generator

    from flight.federation.topologies.node import Node

    NodeState: t.TypeAlias = t.Any

@t.runtime_checkable
class CoordStrategy(t.Protocol):
    def select_workers(
        self, state: NodeState, workers: t.Iterable[Node], rng: Generator
    ) -> t.Sequence[Node]:
        pass
