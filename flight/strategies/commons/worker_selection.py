from collections.abc import Iterable
from typing import cast

from numpy import array
from numpy.random import RandomState
from numpy.typing import NDArray

from flight.federation.topologies.node import Node
from flight.federation.topologies.node import NodeKind


def random_worker_selection(
    children: Iterable[Node],
    participation: float = 1.0,
    probabilistic: bool = False,
    always_include_child_aggregators: bool = True,
    seed: int | None = None,
) -> list[Node]:
    if probabilistic:
        return prob_random_worker_selection(
            children, participation, always_include_child_aggregators, seed
        )
    return fixed_random_worker_selection(children, participation, seed)


def fixed_random_worker_selection(
    children: Iterable[Node], participation: float = 1.0, seed: int | None = None
) -> list[Node]:
    children = array(children)
    rand_state = RandomState(seed)
    num_selected = max(1, int(participation * len(list(children))))

    achildren = cast(NDArray, children)
    selected_children = rand_state.choice(achildren, size=num_selected, replace=False)
    return list(selected_children)


def prob_random_worker_selection(
    children: Iterable[Node],
    participation: float = 1.0,
    always_include_child_aggregators: bool = True,
    seed: int | None = None,
) -> list[Node]:
    rand_state = RandomState(seed)
    selected_children = []
    for child in children:
        if child.kind is NodeKind.WORKER and always_include_child_aggregators:
            selected_children.append(child)
        elif rand_state.uniform() <= participation:
            selected_children.append(child)

    if len(selected_children) == 0:
        achildren = cast(NDArray, children)
        random_child = rand_state.choice(achildren)
        selected_children.append(random_child)

    return selected_children
