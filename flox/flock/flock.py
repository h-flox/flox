from __future__ import annotations

import json
import networkx as nx
import yaml

from pathlib import Path
from typing import Any, Generator, Optional
from uuid import UUID

from flox.flock.node import FlockNode, FlockNodeID, FlockNodeKind


class Flock:
    topo: nx.DiGraph
    node_counter: int

    def __init__(
        self, topo: Optional[nx.DiGraph] = None, _src: Optional[Path | str] = None
    ):
        self.node_counter: int = 0
        self._src = "interactive" if _src is None else _src
        self.leader = None

        if topo is None:
            self.topo = nx.DiGraph()
            self.node_counter += self.topo.number_of_nodes()
        else:
            self.topo = topo
            if not self.validate_topo():
                raise ValueError(
                    "Illegal topology!"
                )  # Expand on this later, the validate function should throw errors.

            found_leader = False
            for idx, data in self.topo.nodes(data=True):
                if data["kind"] is FlockNodeKind.LEADER:
                    if not found_leader:
                        self.leader = FlockNode(
                            idx=idx,
                            kind=data["kind"],
                            globus_compute_endpoint=data["globus_compute_endpoint"],
                            proxystore_endpoint=data["proxystore_endpoint"],
                            # children_idx: Optional[Sequence[FlockNodeID]]
                        )
                        found_leader = True
                    else:
                        raise ValueError(
                            "A legal Flock cannot have more than one leader."
                        )

    def add_node(
        self,
        kind: FlockNodeKind,
        globus_compute_endpoint_id: Optional[UUID] = None,
        proxystore_endpoint_id: Optional[UUID] = None,
    ) -> FlockNodeID:
        if kind is FlockNodeKind.LEADER and self.leader is not None:
            raise ValueError("A leader node has already been established.")

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

    def draw(
        self,
        color_by_kind: bool = True,
        with_labels: bool = True,
        label_color: str = "white",
        prog: str = "dot",
        node_kind_attrs: Optional[dict[FlockNodeKind, str]] = None,
        frameon: bool = False,
    ) -> None:
        pos = nx.nx_agraph.graphviz_layout(self.topo, prog=prog)

        if not color_by_kind:
            nx.draw(self.topo, pos, with_labels=with_labels)
            return

        leader = [self.leader.idx]
        aggregators = list(aggr.idx for aggr in self.aggregators)
        workers = list(worker.idx for worker in self.workers)

        if node_kind_attrs is None:
            node_kind_attrs = {
                FlockNodeKind.LEADER: {"color": "red", "shape": "D", "size": 300},
                FlockNodeKind.AGGREGATOR: {"color": "green", "shape": "s", "size": 300},
                FlockNodeKind.WORKER: {"color": "blue", "shape": "o", "size": 300},
            }

        kinds = [FlockNodeKind.LEADER, FlockNodeKind.AGGREGATOR, FlockNodeKind.WORKER]
        node_sets = [leader, aggregators, workers]
        for kind, nodes in zip(kinds, node_sets):
            nx.draw_networkx_nodes(
                self.topo,
                pos,
                nodes,
                node_color=node_kind_attrs[kind]["color"],
                node_shape=node_kind_attrs[kind]["shape"],
                node_size=node_kind_attrs[kind]["size"],
                label=kind.to_str(),
            )

        nx.draw_networkx_edges(self.topo, pos)
        if with_labels:
            nx.draw_networkx_labels(self.topo, pos, font_color=label_color)

    def validate_topo(self) -> bool:
        # STEP 1: Confirm that there only exists ONE leader in the Flock.
        leaders = []
        for idx, data in self.topo.nodes(data=True):
            if data["kind"] is FlockNodeKind.LEADER:
                leaders.append(idx)
        if len(leaders) != 1:
            return False

        # STEP 2: Confirm that the Flock has a tree topology.
        if not nx.is_tree(self.topo):
            return False

        # TODO: We will also need to confirm that worker nodes are leaves in the tree.
        ...

        return True

    # ================================================================================= #

    @staticmethod
    def from_dict(
        content: dict[str, Any], _src: Optional[Path | str] = None
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

    # TODO: Figure out how to address the issue of JSON requiring string keys for `from_json()`.
    @staticmethod
    def from_json(path: Path | str) -> "Flock":
        """Imports a .json file as a Flock.

        Examples:
            >>> flock = Flock.from_json("my_flock.json")

        Args:
            path (Path | str): Must be a .json file defining a Flock topology.

        Returns:
            An instance of a Flock.
        """
        with open(path, "r") as f:
            content = json.load(f)
        return Flock.from_dict(content, _src=path)

    @staticmethod
    def from_yaml(path: Path | str) -> "Flock":
        """Imports a .yaml file as a Flock.

        Examples:
            >>> flock = Flock.from_yaml("my_flock.yaml")

        Args:
            path (Path | str): Must be a .yaml file defining a Flock topology.

        Returns:
            An instance of a Flock.
        """
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
        # TODO: The leader does NOT need a Globus Compute endpoint.
        key = "globus_compute_endpoint"
        for idx, data in self.topo.nodes(data=True):
            value = data[key]
            if any([value is None, isinstance(value, UUID) == False]):
                return False
        return True

    @property
    def proxystore_ready(self) -> bool:
        key = "proxystore_endpoint"
        for idx, data in self.topo.nodes(data=True):
            value = data[key]
            if any([value is None, isinstance(value, UUID) == False]):
                return False
        return True

    # @property
    # def leader(self):
    #     return self.nodes(by_kind=FlockNodeKind.LEADER)

    @property
    def aggregators(self):
        return self.nodes(by_kind=FlockNodeKind.AGGREGATOR)

    @property
    def workers(self) -> Generator[FlockNode]:
        return self.nodes(by_kind=FlockNodeKind.WORKER)

    def nodes(self, by_kind: Optional[FlockNodeKind] = None) -> Generator[FlockNode]:
        for idx, data in self.topo.nodes(data=True):
            if by_kind is not None and data["kind"] != by_kind:
                continue
            yield FlockNode(
                idx=idx,
                kind=data["kind"],
                globus_compute_endpoint=data["globus_compute_endpoint"],
                proxystore_endpoint=data["proxystore_endpoint"],
                # children_idx: Optional[Sequence[FlockNodeID]]
            )

    # ================================================================================= #

    def __repr__(self):
        return f"Flock(`{self._src}`)"