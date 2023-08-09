from __future__ import annotations

import json
import networkx as nx
import yaml

from pathlib import Path
from typing import Any, Optional, overload, Sequence
from uuid import UUID

from flox.flock.node import FlockNodeID, FlockNodeKind


class Flock:
    topo: nx.DiGraph
    node_counter: int

    def __init__(
        self, topo: Optional[nx.DiGraph] = None, _src: Optional[str | Path] = None
    ):
        self.node_counter: int = 0
        self._src = "interactive" if _src is None else _src

        if topo is None:
            self.topo = nx.DiGraph()
            self.node_counter += self.topo.number_of_nodes()
        else:
            self.topo = topo

    def add_node(
        self,
        kind: FlockNodeKind,
        globus_compute_endpoint_id: Optional[UUID] = None,
        proxystore_endpoint_id: Optional[UUID] = None,
    ) -> FlockNodeID:
        idx = self.node_counter
        self.topo.add_node(
            idx,
            kind=kind,
            globus_compute_endpoint_id=globus_compute_endpoint_id,
            proxystore_endpoint_id=proxystore_endpoint_id,
        )
        self.node_counter += 1
        return FlockNodeID(idx)

    def add_edge(self, idx1: FlockNodeID, idx2: FlockNodeID, **attr):
        if any([idx1 not in self.topo.nodes, idx2 not in self.topo.nodes]):
            raise ValueError(
                "`Flock` does not support adding edges between nodes that do not already exist. "
                "Try adding each node first."
            )
        self.topo.add_edge(idx1, idx2, **attr)

    def _validate_topo(self) -> bool:
        return nx.is_tree(self.topo)

    @staticmethod
    def from_dict(
        content: dict[str, Any], _src: Optional[str | Path] = None
    ) -> "Flock":
        topo = nx.DiGraph()
        required_attrs: set[str] = {
            "kind",
            "globus_compute_endpoint",
            "proxystore_endpoint",
            "children",
        }

        # STEP 1: Add the nodes with their attributes --- ignore children for now.
        for node_idx, values in content.items():
            for attr in required_attrs:
                if attr not in values:
                    raise ValueError(
                        f"Node {node_idx} does not have required attribute: `{attr}`."
                    )

            topo.add_node(
                node_idx,
                kind=FlockNodeKind.from_str(values["kind"]),
                globus_compute_endpoint=values["globus_compute_endpoint"],
                proxystore_endpoint=values["proxystore_endpoint"],
            )
            for extra_attr in set(values) - required_attrs:
                topo.nodes[node_idx][extra_attr] = values[extra_attr]

        # STEP 2: Add the edges from the children attribute.
        for node_idx, values in content.items():
            for child in values["children"]:
                topo.add_edge(node_idx, child)

        return Flock(topo=topo, _src=_src)

    @staticmethod
    def from_json(path: Path) -> "Flock":
        with open(path, "r") as f:
            content = json.load(f)
        return Flock.from_dict(content, _src=path)

    @staticmethod
    def from_yaml(path: Path) -> "Flock":
        with open(path, "r") as f:
            content = yaml.safe_load(f)
        return Flock.from_dict(content, _src=path)

    # ================================================================================= #

    @property
    def globus_compute_ready(self) -> bool:
        """
        True if the Flock instance has all necessary endpoints to be run across
        Globus Compute; False otherwise.
        """
        key = "globus_compute_endpoint"
        for node in self.topo.nodes:
            value = self.topo.nodes[node][key]
            if any([value is None, isinstance(value, UUID) == False]):
                return False
        return True

    @property
    def proxystore_ready(self) -> bool:
        key = "proxystore_endpoint"
        for node in self.topo.nodes:
            value = self.topo.nodes[node][key]
            if any([value is None, isinstance(value, UUID) == False]):
                return False
        return True

    # ================================================================================= #

    def __repr__(self):
        return f"Flock(`{self._src}`)"
