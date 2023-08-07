import networkx as nx
import uuid
from typing import Optional


def make_flock(num_nodes: int, seed=None) -> tuple[nx.DiGraph, dict]:
    flock = nx.random_tree(num_nodes, create_using=nx.DiGraph, seed=seed)
    aggrs = [
        node
        for node in flock.nodes()
        if flock.out_degree(node) > 0 and flock.in_degree(node) > 0
    ]
    workers = [
        node
        for node in flock.nodes()
        if flock.out_degree(node) == 0 and flock.in_degree(node) == 1
    ]

    for aggr in aggrs:
        flock.nodes[aggr]["kind"] = "aggr"

    for worker in workers:
        flock.nodes[worker]["kind"] = "worker"

    return flock, {"aggrs": aggrs, "workers": workers}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    flock, info = make_flock(8, seed=123)
    aggrs = info["aggrs"]
    workers = info["workers"]

    print(flock.nodes(data=True))
    pos = nx.nx_agraph.graphviz_layout(flock, prog="dot")
    nx.draw_networkx_nodes(flock, pos, nodelist=aggrs)
    plt.show()
