import networkx as nx

from flox.flock import Flock


def random_flock(num_nodes: int, seed: int | None = None) -> Flock:
    """Generates a random Flock network.

    Args:
        num_nodes (int): ...
        seed (int | None): ...

    Returns:
        A random Flock using ``networkx.random_tree()``.
    """
    # TODO: Finish this and create a test.
    tree = nx.random_tree(n=num_nodes, seed=seed, create_using=nx.DiGraph)
    for node in tree.nodes():
        for attr in Flock.required_attrs:
            tree.nodes[node][attr] = None
    flock = Flock(tree)
    return flock
