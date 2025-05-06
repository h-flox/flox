from __future__ import annotations

import math
import typing as t

import networkx as nx
import numpy as np

from .node import Node, NodeID, NodeKind
from .topology import Topology


def flat_topology(n: int, /, **kwargs) -> Topology:
    """Creates a flat `Topology` where Workers are directly connected to
    the Coordinator.

    Args:
        n (int): Number of worker nodes.
        **kwargs: Keyword arguments that are passed onto every node in the
            created `Topology`.

    Returns:
        A flat topology with `n + 1` total nodes.
    """
    edges = []
    nodes = [Node(0, "coordinator", **kwargs)]

    for i in range(1, n + 1):
        nodes.append(Node(i, "worker", **kwargs))
        edges.append((0, i))

    return Topology(nodes, edges)


def hierarchical_topology(
    n: int,
    aggr_shape: t.Sequence[int] | None = None,
    rng: np.random.Generator | int | None = None,
) -> Topology:
    """
    Constructs a hierarchical topology.

    This is a flexible function that allows users to rapidly produce hierarchical
    `Topology` object by setting the number of works (`n`) and the shape of
    intermediate aggregators (`aggr_shape`). The `aggr_shape` argument is a
    `typing.Sequence` of integers in *ascending order* that defines the layers of
    intermediate aggregators in the resulting `Topology`.

    For instance consider `aggr_shape=[2, 4]` and `n=10`. These arguments will
    result in a topology with a single coordinator with connections to `2`
    aggregators, these `2` aggregators will have random connections covering `4`
    aggregators, and these `4` aggregators will have random connections covering
    `10` workers.

    This results in a topology similar to the following:

    ```mermaid
    flowchart

        coord-->a-1.1
        coord-->a-1.2

        a-1.1-->a-2.1
        a-1.1-->a-2.2

        a-1.2-->a-2.3
        a-1.2-->a-2.4

        a-2.1-->w-1
        a-2.1-->w-2
        a-2.1-->w-3

        a-2.2-->w-4
        a-2.2-->w-5

        a-2.3-->w-6
        a-2.3-->w-7

        a-2.4-->w-8
        a-2.4-->w-9
        a-2.4-->w-10
    ```

    Args:
        n (int): The number of worker nodes.
        aggr_shape (typing.Sequence[int] | None): The shape of intermediate aggregator
            layers. If this argument is `None`, then `flat_topology` is called with `n`.
        rng (numpy.random.Generator | int | None): Random generator for repeatability.
            This argument can be a generator (`numpy.random.Generator`),
            a random seed (`int`) or pseudorandom (`None`).

    Returns:
        A hierarchical `Topology` instance.

    Notes:
        This function behaves randomly in terms of how the nodes are connected
        in a `Topology`. Be mindful of this. To ensure reproducibility, take
        advantage of the `rng` argument.
    """
    # If no `aggr_shape` argument is provided, then we simply return the result
    # of the `flat_topology` function since the user specified no intermediate
    # aggregators.
    if aggr_shape is None:
        return flat_topology(n)

    # In the case the user provided `aggr_shape`, let's first confirm that the
    # user provided a legal value for `aggr_shape`.
    for i in range(len(aggr_shape) - 1):
        u, v = aggr_shape[i], aggr_shape[i + 1]
        if u > v:
            raise ValueError(
                "Argument `aggr_shape` must have ascending values "
                "(i.e., no value can be larger than the preceding value). "
                "This is because, in legal topologies in Flight, two nodes "
                "at the same depth in the tree cannot share a child node."
            )

        conditions = [not 0 < u <= n, not 0 < v <= n]
        if any(conditions):
            raise ValueError(f"Values in `aggr_shape` must be in range (0, {n=}].")

    match rng:
        case np.random.Generator():
            rng = rng
        case int() | None:
            rng = np.random.default_rng(rng)
        case _:
            raise ValueError("Illegal value given for argument `rng`.")

    def _choose_parents(
        _tree: nx.DiGraph,
        _children: t.Sequence[NodeID],
        _parents: t.Sequence[NodeID],
        _rng: np.random.Generator,
    ):
        children_without_parents = [child for child in _children]

        for parent in _parents:
            child = _rng.choice(children_without_parents)
            children_without_parents.remove(child)
            tree.add_edge(parent, child)

        for child in children_without_parents:
            parent = _rng.choice(_parents)
            tree.add_edge(parent, child)

    # Begin constructing the tree to build the `Topology` instance.
    coord_idx = 0
    tree: nx.DiGraph = nx.DiGraph()
    tree.add_node(
        coord_idx,
        kind=NodeKind.COORDINATOR,
        globus_comp_id=None,
        proxystore_id=None,
    )

    workers: list[NodeID] = []
    for i in range(1, n + 1):
        tree.add_node(
            i,
            kind=NodeKind.WORKER,
            globus_comp_id=None,
            proxystore_id=None,
        )
        workers.append(i)

    aggr_idx = 1 + n
    last_aggrs = [coord_idx]
    for num_aggrs in aggr_shape:
        if not 0 < num_aggrs <= n:
            raise ValueError(
                "Value for number of aggregators in 'middle' tier must be nonzero and "
                "no greater than the number of workers."
            )

        curr_aggrs = []
        for _ in range(num_aggrs):
            tree.add_node(
                aggr_idx,
                kind=NodeKind.AGGREGATOR,
                globus_comp_id=None,
                proxystore_id=None,
            )
            curr_aggrs.append(aggr_idx)
            aggr_idx += 1

        _choose_parents(tree, curr_aggrs, last_aggrs, rng)
        last_aggrs = curr_aggrs

    _choose_parents(tree, workers, last_aggrs, rng)

    return Topology.from_networkx(tree)


def balanced_topology(b: int, h: int) -> Topology:
    """
    Creates a *balanced tree* for a `Topology`.

    Args:
        b (int): Branching factor (how many children non-leave nodes will have).
        h (int): Height of the tree.

    Returns:
        A `Topology` instance with nodes organized as a balanced tree.
    """
    tree = nx.balanced_tree(b, h, create_using=nx.DiGraph)
    for node_id, node_data in tree.nodes(data=True):
        num_parents = len(list(tree.predecessors(node_id)))
        num_children = len(list(tree.successors(node_id)))

        if num_parents == 0:
            node_data["kind"] = NodeKind.COORDINATOR
        elif num_children == 0:
            node_data["kind"] = NodeKind.WORKER
        else:
            node_data["kind"] = NodeKind.AGGREGATOR

        node_data["globus_comp_id"] = None
        node_data["proxystore_id"] = None

    return Topology.from_networkx(tree)


def balanced_topology_by_leaves(
    leaves: int,
    height: int,
    rounding: t.Literal["round", "floor", "ceil"] = "round",
) -> Topology:
    r"""
    Creates a `Topology`` with a balanced tree topology with a (roughly) fixed number
    of leaves (i.e., workers).

    By default, `networkx` provides the `balanced_tree` function which generates a
    balanced tree using the branching factor and the height of the tree. This function
    wraps that function and computes the branching factor using
    $b = \lfloor l^{1/h} \rfloor where $l$ is the number of leaves and $h$ is the
    height.

    Args:
        leaves (int): Approximate number of leaves in the resulting tree (see note).
        height (int): Height of the tree.
        rounding (t.Literal): How to round the branching factor.

    Returns:
        The `Topology` instance with the constructed, balanced tree.

    Notes:
        Because the calculation for $b$ (described above) is not always going to result
        in an integer, this function will use the floor of $l^{1/h}$. Unless you are
        wise about your parameters for `leaves` and `height`, you will have more
        leaves than originally specified. So, be mindful of this.

        It is worth being mindful of the values used for the `height` and `leaves`
        arguments. Multiples of two are more reliable. It is common to result in
        having many more leaves than you might specify. So be mindful of this when
        using this function.
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
            raise ValueError("Illegal value for arg `rounding`.")

    tree = nx.balanced_tree(r, height, create_using=nx.DiGraph)

    for idx in tree.nodes():
        parents = list(tree.predecessors(idx))
        children = list(tree.successors(idx))

        if len(parents) == 0:
            tree.nodes[idx]["kind"] = NodeKind.COORDINATOR
        elif len(children) == 0:
            tree.nodes[idx]["kind"] = NodeKind.WORKER
        else:
            tree.nodes[idx]["kind"] = NodeKind.AGGREGATOR

        tree.nodes[idx]["globus_comp_id"] = None
        tree.nodes[idx]["proxystore_id"] = None

    return Topology.from_networkx(tree)
