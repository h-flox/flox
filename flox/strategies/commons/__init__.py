"""
Provides common, basic methods that serve as "building blocks" for making strategies.
This is because there is often much overlap in terms of the functionality a `Strategy`
will use for one of its callbacks. This just helps make code more reusable and simplify
the process of kick-starting research efforts into defining novel Strategies.
"""

from numpy.random import RandomState
from typing import Iterable

from flox.flock import FlockNode, FlockNodeKind


def random_worker_selection(
    children: list[FlockNode],
    participation: float = 1.0,
    probabilistic: bool = False,
    always_include_child_aggregators: bool = True,
    seed: int = None,
) -> list[FlockNode]:
    """

    Args:
        children ():
        participation ():
        probabilistic ():
        always_include_child_aggregators ():
        seed ():

    Returns:

    """
    if probabilistic:
        return prob_random_worker_selection(
            children, participation, always_include_child_aggregators, seed
        )
    return fixed_random_worker_selection(children, participation, seed)


def fixed_random_worker_selection(
    children: list[FlockNode],
    participation: float = 1.0,
    seed: int = None,
) -> list[FlockNode]:
    """

    Args:
        children ():
        participation ():
        seed ():

    Returns:

    """
    rand_state = RandomState(seed)
    num_selected = min(1, int(participation) * len(list(children)))
    selected_children = rand_state.choice(children, size=num_selected, replace=False)
    return list(selected_children)


def prob_random_worker_selection(
    children: list[FlockNode],
    participation: float = 1.0,
    always_include_child_aggregators: bool = True,
    seed: int = None,
) -> list[FlockNode]:
    """

    Args:
        children ():
        participation ():
        always_include_child_aggregators ():
        seed ():

    Returns:

    """
    rand_state = RandomState(seed)
    selected_children = []
    for child in children:
        if child.kind is FlockNodeKind.WORKER and always_include_child_aggregators:
            selected_children.append(child)
        elif rand_state.uniform() < participation:
            selected_children.append(child)

    if len(selected_children) == 0:
        random_child = rand_state.choice(children)
        selected_children.append(random_child)

    return selected_children
