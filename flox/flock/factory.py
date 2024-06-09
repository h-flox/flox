from __future__ import annotations

import math
import random
import typing as t

import networkx as nx

from flox.flock import Flock, NodeKind

if t.TYPE_CHECKING:
    from flox.flock import FlockNode


def create_standard_flock(num_workers: int, **edge_attrs) -> Flock:
    flock = Flock()
    flock.leader = flock.add_node("leader")
    for _ in range(num_workers):
        worker = flock.add_node("worker")
        flock.add_edge(flock.leader.idx, worker.idx, **edge_attrs)
    return flock


def _choose_parents(tree: nx.DiGraph, children, parents):
    children_without_parents = [child for child in children]

    for parent in parents:
        child = random.choice(children_without_parents)
        children_without_parents.remove(child)
        tree.add_edge(parent, child)

    for child in children_without_parents:
        parent = random.choice(parents)
        tree.add_edge(parent, child)


def create_hierarchical_flock(
    workers: int, aggr_shape: t.List[int] | None = None, return_nx: bool = False
) -> Flock | nx.DiGraph:
    client_idx = 0
    flock = nx.DiGraph()
    flock.add_node(
        client_idx,
        kind=NodeKind.LEADER,
        proxystore_endpoint=None,
        globus_compute_endpoint=None,
    )
    worker_nodes = []
    for i in range(workers):
        idx = i + 1
        flock.add_node(
            idx,
            kind=NodeKind.WORKER,
            proxystore_endpoint=None,
            globus_compute_endpoint=None,
        )
        worker_nodes.append(idx)

    if aggr_shape is None:
        for worker in worker_nodes:
            flock.add_edge(client_idx, worker)
        if return_nx:
            return flock
        return Flock(flock)

    # Validate the values of the `aggr_shape` argument.
    for i in range(len(aggr_shape) - 1):
        v0, v1 = aggr_shape[i], aggr_shape[i + 1]
        if v0 > v1:
            raise ValueError(
                "Argument `aggr_shape` must have ascending values "
                "(i.e., no value can be larger than the preceding value)."
            )
        if not 0 < v0 <= workers or not 0 < v1 <= workers:
            raise ValueError(
                f"Values in `aggr_shape` must be in range (0, `{workers=}`]."
            )

    aggr_idx = 1 + workers
    last_aggrs = [client_idx]
    for num_aggrs in aggr_shape:
        if not 0 < num_aggrs <= workers:
            raise ValueError(
                "Value for number of aggregators in 'middle' tier must be nonzero and "
                "no greater than the number of workers."
            )

        curr_aggrs = []
        for aggr in range(num_aggrs):
            flock.add_node(
                aggr_idx,
                kind=NodeKind.AGGREGATOR,
                proxystore_endpoint=None,
                globus_compute_endpoint=None,
            )
            curr_aggrs.append(aggr_idx)
            aggr_idx += 1

        _choose_parents(flock, curr_aggrs, last_aggrs)
        last_aggrs = curr_aggrs

    _choose_parents(flock, worker_nodes, last_aggrs)

    if return_nx:
        return flock
    return Flock(flock)


def created_balanced_hierarchical_flock(branching_factor: int, height: int):
    tree = nx.balanced_tree(branching_factor, height, create_using=nx.DiGraph)
    gce = "globus_compute_endpoint"
    pse = "proxystore_endpoint"
    for node_id, node_data in tree.nodes(data=True):
        num_parents = len(list(tree.predecessors(node_id)))
        num_children = len(list(tree.successors(node_id)))

        if num_parents == 0:
            node_data["kind"] = NodeKind.LEADER
        elif num_children == 0:
            node_data["kind"] = NodeKind.WORKER
        else:
            node_data["kind"] = NodeKind.AGGREGATOR
        node_data[gce] = None
        node_data[pse] = None

    return Flock(tree)


def created_balanced_hierarchical_flock_by_leaves(
    leaves: int,
    height: int,
    rounding: t.Literal["round", "floor", "ceil"] = "round",
) -> Flock:
    r"""
    Creates a Flock with a balanced tree topology with a (roughly) fixed number of leaves.

    By default, `networkx` provides the `balanced_tree` function which generates a balanced
    tree using the branching factor and the height of the tree. This function wraps that function
    and computes the branching factor using $b = \lfloor l^{1/h} \rfloor where $l$ is the number
    of leaves and $h$ is the height.

    Notes:
        Because the calculation for $b$ (described above) is not always going to result in an
        integer, this function will use the floor of $l^{1/h}$. Unless you are wise about your
        parameters for `leaves` and `height`, you will have more leaves than originally specified.
        So, be mindful of this.

    Args:
        leaves (int): Approximate number of leaves in the resulting tree (see note).
        height (int): Height of the tree.
        rounding (t.Literal): How to round the branching factor.

    Notes:
        It is worth being mindful of the values used for the `height` and `leaves` arguments.
        Multiples of two are more reliable. It is common to result in having many more leaves than
        you might specify. So be mindful of this when using this function.

    Returns:
        The Flock instance with the constructed, balanced tree.
    """
    if leaves < 1:
        raise ValueError("Value for arg `leaves` must be at least 1.")

    branching_factor = leaves ** (1 / height)
    match rounding:
        case "round":
            r = round(branching_factor)
        case "floor":
            r = math.floor(branching_factor)
        case "ceil":
            r = math.ceil(branching_factor)
        case _:
            raise ValueError(f"Illegal value for arg `rounding`.")

    tree = nx.balanced_tree(r, height, create_using=nx.DiGraph)

    for idx in tree.nodes():
        parents = list(tree.predecessors(idx))
        children = list(tree.successors(idx))

        if len(parents) == 0:
            tree.nodes[idx]["kind"] = NodeKind.LEADER
        elif len(children) == 0:
            tree.nodes[idx]["kind"] = NodeKind.WORKER
        else:
            tree.nodes[idx]["kind"] = NodeKind.AGGREGATOR

        tree.nodes[idx]["globus_compute_endpoint"] = None
        tree.nodes[idx]["proxystore_endpoint"] = None

    return Flock(tree)


def from_yaml():
    pass


def from_dict(topo: dict[str, t.Any]):
    pass


def from_list(topo: list[FlockNode | dict[str, t.Any]]):
    pass


def from_json():
    pass
