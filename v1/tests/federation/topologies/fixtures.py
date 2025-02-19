from __future__ import annotations

import collections as c

import pytest

GraphFixture = c.namedtuple("GraphFixture", ["graph", "stats"])


@pytest.fixture
def two_tier_graph() -> GraphFixture:
    workers = list(range(1, 11))
    graph = {
        0: {
            "kind": "coordinator",
            "globus_comp_id": None,
            "proxystore_id": None,
            "extra": None,
            "children": workers,
        }
    }
    for i in workers:
        graph[i] = {
            "kind": "worker",
            "globus_comp_id": None,
            "proxystore_id": None,
            "extra": None,
            "children": [],
        }

    stats = dict(num_nodes=11, num_aggrs=0, num_workers=10)
    return GraphFixture(graph, stats)


@pytest.fixture
def three_tier_graph() -> GraphFixture:
    data = {
        0: {
            "kind": "coordinator",
            "globus_comp_id": None,
            "proxystore_id": None,
            "extra": None,
            "children": [1, 2],
        },
        1: {
            "kind": "aggregator",
            "globus_comp_id": None,
            "proxystore_id": None,
            "extra": None,
            "children": [3, 4, 5],
        },
        2: {
            "kind": "aggregator",
            "globus_comp_id": None,
            "proxystore_id": None,
            "extra": None,
            "children": [6, 7, 8],
        },
    }
    for node in (3, 4, 5, 6, 7, 8):
        data[node] = {
            "kind": "worker",
            "globus_comp_id": None,
            "proxystore_id": None,
            "extra": None,
            "children": [],
        }
    stats = dict(num_nodes=9, num_aggrs=2, num_workers=6)
    return GraphFixture(data, stats)
