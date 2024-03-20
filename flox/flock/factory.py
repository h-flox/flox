from __future__ import annotations

import typing

import networkx as nx

from flox.flock import Flock

if typing.TYPE_CHECKING:
    from flox.flock import FlockNode


def create_standard_flock(num_workers: int, **edge_attrs) -> Flock:
    flock = Flock()
    flock.leader = flock.add_node("leader")
    for _ in range(num_workers):
        worker = flock.add_node("worker")
        flock.add_edge(flock.leader.idx, worker.idx, **edge_attrs)
    return flock


def create_hier_flock(branch_factor: int, height: int, create_using=None, **edge_attrs):
    tree = nx.generators.balanced_tree(branch_factor, height, create_using=nx.DiGraph)
    flock = Flock()
    flock.leader = flock.add_node("leader")

    for node in tree.nodes():
        parents = list(tree.predecessors(node))
        children = list(tree.successors(node))

        if len(parents) == 0:
            continue

        elif len(children) > 0:
            aggr = flock.add_node("aggregator")
            flock.add_edge(flock.leader.idx, aggr.idx, **edge_attrs)

        else:
            assert len(parents) == 1
            worker = flock.add_node("worker")
            flock.add_edge(parents[0], worker.idx, **edge_attrs)

    return flock


def from_yaml():
    pass


def from_dict(topo: dict[str, typing.Any]):
    pass


def from_list(topo: list[FlockNode | dict[str, typing.Any]]):
    pass


def from_json():
    pass
