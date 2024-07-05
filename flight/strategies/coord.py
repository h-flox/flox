from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from numpy.random import Generator

    from ..federation.topologies.node import Node

    CoordState: t.TypeAlias = t.Any


class CoordStrategy(t.Protocol):
    def select_workers(
        self, state: CoordState, workers: t.Iterable[Node], rng: Generator
    ) -> t.Sequence[Node]:
        pass
