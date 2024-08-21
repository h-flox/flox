from .topo import Topology


def flat_topology(n: int) -> Topology:
    workers = list(range(1, n + 1))
    data = {0: dict(kind="coordinator", children=workers)}
    data.update({i: dict(kind="worker", children=[]) for i in workers})
    return Topology.from_dict(data)
