from collections.abc import Iterable

from numpy import array
from numpy.random import Generator, default_rng

from flight.federation.topologies.node import Node, NodeKind


def random_worker_selection(
    children: Iterable[Node],
    participation: float = 1.0,
    probabilistic: bool = False,
    always_include_child_aggregators: bool = True,
    rng: Generator | None = None,
) -> list[Node]:
    """
    General call for worker selection that will then choose from probabilistic or
    fixed selection.

    Args:
        children (Iterable[Node]): Children to be evaluated for worker selection.
        participation (float, optional): Controls the level of participation each node
            contributes. Defaults to 1.0.
        probabilistic (bool, optional): Decider for whether probabilistic (True), or
            fixed (False) selection will be used. Defaults to False.
        always_include_child_aggregators (bool, optional): In probabilistic selection,
            ensures whether all worker nodes are included. Defaults to True.
        rng (Generator | None, optional): RNG object used for randomness,
            numpy.random.default_rng will be used if None. Defaults to None.

    Returns:
        list[Node]: The selected worker nodes.
    """
    if rng is None:
        rng = default_rng()
    if probabilistic:
        return prob_random_worker_selection(
            children, rng, participation, always_include_child_aggregators
        )
    return fixed_random_worker_selection(children, rng, participation)


def fixed_random_worker_selection(
    children: Iterable[Node],
    rng: Generator,
    participation: float = 1.0,
) -> list[Node]:
    """
    The worker selection used when `probabilistic` arg is set to false. This worker
    selection is entirely random based on 'rng'

    Args:
        children (Iterable[Node]): Children to be evaluated for worker selection.
        rng (Generator): RNG object used for randomness.
        participation (float, optional): Controls the level of participation each node
            contributes. Defaults to 1.0.

    Returns:
        list[Node]: The selected worker nodes.
    """
    children = list(children)
    children = array(children)
    num_selected = max(1, int(participation * len(list(children))))
    selected_children = rng.choice(children, size=num_selected, replace=False)
    return list(selected_children)


def prob_random_worker_selection(
    children: Iterable[Node],
    rng: Generator,
    participation: float = 1.0,
    always_include_child_aggregators: bool = True,
) -> list[Node]:
    """
    The worker selection used when probabilistic is true. This worker selection is
    probabilistic and therefore in most cases will select workers in order.

    Args:
        children (Iterable[Node]): Children to be evaluated for worker selection.
        rng (Generator): RNG object used for randomness.
        participation (float, optional): Acts as a probability marker for whether
            to include a node. Defaults to 1.0.
        always_include_child_aggregators (bool, optional): Ensures whether worker
            nodes are included. Defaults to True.

    Returns:
        list[Node]: The selected worker nodes.
    """
    selected_children = []
    for child in children:
        if child.kind is NodeKind.WORKER and always_include_child_aggregators:
            selected_children.append(child)
        elif rng.uniform() <= participation:
            selected_children.append(child)

    if len(selected_children) == 0:
        random_child = rng.choice(array(children))
        selected_children.append(random_child)

    return selected_children
