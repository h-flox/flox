from __future__ import annotations

import typing

from flox.flock import Flock

if typing.TYPE_CHECKING:
    from flox.flock import FlockNode


def create_two_tier_flock(num_workers: int, **edge_attrs) -> Flock:
    flock = Flock()
    flock.leader = flock.add_node("leader")
    for i in range(num_workers):
        worker = flock.add_node("worker")
        flock.add_edge(flock.leader.idx, worker.idx, **edge_attrs)
    return flock


def from_yaml():
    pass


def from_dict(topo: dict[str, typing.Any]):
    pass


def from_list(topo: list[FlockNode | dict[str, typing.Any]]):
    pass


def from_json():
    pass
