from v1.flight import Topology
from v1.flight import find_relevant_nodes


def test_find_intermediate_aggregators():
    data = {0: {"kind": "coordinator", "children": [idx for idx in range(1, 10 + 1)]}}
    data.update({idx: {"kind": "worker"} for idx in range(1, 10 + 1)})
    flat_topo = Topology.from_dict(data)

    for idx in range(1, 10 + 1):
        print(f"{idx=}")
        selected_nodes = [flat_topo[idx]]
        intermediate_aggrs = find_relevant_nodes(
            flat_topo,
            selected_nodes,
        )
        assert len(intermediate_aggrs) == 0

    workers = range(3, 10 + 1)
    data = {
        0: {"kind": "coordinator", "children": [1, 2, 3]},
        1: {"kind": "aggregator", "children": [4, 5, 6, 7]},
        2: {"kind": "aggregator", "children": [8, 9, 10]},
    }
    data.update({idx: {"kind": "worker"} for idx in workers})
    hier_topo = Topology.from_dict(data)
    correct_answers = {
        3: [],
        4: [1],
        5: [1],
        6: [1],
        7: [1],
        8: [2],
        9: [2],
        10: [2],
    }
    for w in workers:
        intermediate_aggrs = find_relevant_nodes(
            hier_topo,
            [hier_topo[w]],
        )
        assert len(intermediate_aggrs) == len(correct_answers[w])
