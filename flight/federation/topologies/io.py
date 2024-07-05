from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    import pathlib

    import numpy as np

    from .topo import Topology


def from_yaml(path: pathlib.Path | str) -> Topology:
    pass


def from_json(path: pathlib.Path | str) -> Topology:
    pass


def from_edgelist(path: pathlib.Path | str) -> Topology:
    pass


def from_adj_list(adj_list: t.Mapping[t.Hashable, t.Sequence[t.Hashable]]) -> Topology:
    pass


def from_adj_matrix(matrix: np.ndarray) -> Topology:
    pass
