from __future__ import annotations

# import io
import typing as t

import flight.federation.topologies.io as io

if t.TYPE_CHECKING:
    import networkx as nx


def validate_graph(graph: nx.Graph):
    pass


class Topology:
    graph: nx.DiGraph

    def __init__(self, graph: nx.DiGraph):
        pass

    from_adj_list = staticmethod(io.from_adj_list)
    from_adj_matrix = staticmethod(io.from_adj_matrix)
    from_edgelist = staticmethod(io.from_edgelist)
    from_json = staticmethod(io.from_json)
    from_yaml = staticmethod(io.from_yaml)
