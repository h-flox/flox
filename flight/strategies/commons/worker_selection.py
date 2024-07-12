from collections.abc import Iterable
from typing import cast

from numpy import array
from numpy.random import Generator, RandomState, default_rng
from numpy.typing import NDArray

from flight.federation.topologies.node import Node, NodeKind


def random_worker_selection(
    children: Iterable[Node],
    participation: float = 1.0,
    probabilistic: bool = False,
    always_include_child_aggregators: bool = True,
    rng: Generator | None = None,
) -> list[Node]:
    if rng is None:
        rng = default_rng()
    if probabilistic:
        return prob_random_worker_selection(
            children, rng, participation, always_include_child_aggregators
        )
    return fixed_random_worker_selection(children, rng, participation)


def fixed_random_worker_selection(
    children: Iterable[Node], rng: Generator, participation: float = 1.0
) -> list[Node]:
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
