from __future__ import annotations

import typing as t

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes

from .node import NodeKind

if t.TYPE_CHECKING:
    from .topo import Topology


_PROGS = [
    "dot",
    "neato",
    "fdp",
    "sfdp",
    "circo",
    "twopi",
    "nop",
    "nop2",
    "osage",
    "patchwork",
]


def draw(
    topo: Topology,
    color_by_kind: bool = True,
    with_labels: bool = True,
    label_color: str = "white",
    prog: str = "circo",
    node_kind_attrs: dict[NodeKind, dict[str, t.Any]] | None = None,
    show_axis_border: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """
    Draws the topos using Matplotlib. The nodes are organized as a tree with the proper
    hierarchy based on depth from the Leader node (root).

    Args:
        topo (Topology): The topology to draw.
        color_by_kind (bool): Color nodes by kind, if True.
        with_labels (bool): Display labels of nodes, if True.
        label_color (str): Color of labels.
        prog (str): How the topology is organized. Leave alone for the default behavior
            of displaying it as a tree. This is passed into the `prog` argument for
            ``networkx.nx_agraph.graphviz_layout()``.
        node_kind_attrs (): Determines how node attributes should be plotted.
            By default, nodes will be colored and marked by kind.
        show_axis_border (bool): Show the border along the axis if True;
            defaults to False.
        ax (Axes | None): Axes object to draw onto. If none is provided, then one
            will be created.

    Returns:
        Axes object that was drawn onto.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if not show_axis_border:
        ax.axis("off")

    # TODO: We may want to remove this as a requirement. It produces nice "tree"
    #       positions of the nodes. But it introduces a pretty restrictive dependency.
    if prog in _PROGS:
        pos = nx.nx_pydot.pydot_layout(topo.graph, prog=prog)
    else:
        pos = nx.spring_layout(topo.graph)

    if not color_by_kind:
        nx.draw(topo.graph, pos, with_labels=with_labels, ax=ax)
        return ax

    if topo.coordinator is None:
        raise ValueError(
            "There is no leader in the Flock. This is likely because no topology "
            "has been created via interactive mode."
        )

    coord = [topo.coordinator.idx]
    aggregators = list(aggr.idx for aggr in topo.aggregators)
    workers = list(worker.idx for worker in topo.workers)

    if node_kind_attrs is None:
        node_kind_attrs = {
            NodeKind.COORD: {"color": "red", "shape": "D", "size": 300},
            NodeKind.AGGR: {"color": "green", "shape": "s", "size": 300},
            NodeKind.WORKER: {"color": "blue", "shape": "o", "size": 300},
        }

    kinds = [NodeKind.COORD, NodeKind.AGGR, NodeKind.WORKER]
    node_sets = [coord, aggregators, workers]
    for kind, nodes in zip(kinds, node_sets):
        if len(nodes) == 0:
            continue
        nx.draw_networkx_nodes(
            topo.graph,
            pos,
            nodes,
            node_color=node_kind_attrs[kind]["color"],
            node_shape=node_kind_attrs[kind]["shape"],
            node_size=node_kind_attrs[kind]["size"],
            label=kind.value,
            ax=ax,
        )

    nx.draw_networkx_edges(topo.graph, pos)
    if with_labels:
        nx.draw_networkx_labels(topo.graph, pos, font_color=label_color, ax=ax)

    return ax
