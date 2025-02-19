from v1.flight.topologies.node import Node, NodeKind
from v1.flight.topologies.topo import Topology


def main():
    nodes = [
        Node(idx=0, kind=NodeKind.COORD),
        Node(idx=1, kind=NodeKind.WORKER),
        Node(idx=2, kind=NodeKind.WORKER),
        Node(idx=3, kind=NodeKind.WORKER),
    ]
    edges = [(0, 1), (0, 2), (0, 3)]
    topo = Topology(nodes, edges)
    for node in topo.nodes(kind="coordinator"):
        print(node)

    print(f"{len(topo)=}")
    print(f"{topo.number_of_nodes('worker')=}")
    print(f"{topo.number_of_nodes('coordinator')=}")


if __name__ == "__main__":
    main()
