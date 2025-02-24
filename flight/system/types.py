import typing as t

NodeID: t.TypeAlias = int
"""ID of a `Node`."""

Edge: t.TypeAlias = tuple[NodeID, NodeID]
"""
Type alias that represents an edge/link/connection between nodes
in the graph defining the Flight `Topology`.

Example:
    ```python
    u: NodeID = 0
    v: NodeID = 1
    link: Edge = (u, v)
    ```
"""

GraphDict: t.TypeAlias = dict[NodeID, dict[str, t.Any]]
"""
A dictionary where each top-level key is the node ID (`str` or `int`)
and the values are Mappings (e.g., `dict` or `collection.OrderedDict`
objects) with `str` keys for each input into the `Node` class and the
child Node IDs.

Example:
    ```python
    data: GraphDict = {
        0: {
            'kind': 'coordinator',
            'globus_comp_id': None, # or UUID
            'proxystore_id': None, # or UUID
            'children': [1, 2],
            'extra: None,
        },
        1: {
            'kind': 'worker',
            'globus_comp_id': None, # or UUID
            'proxystore_id': None, # or UUID
            'children': [],
            'extra: None,
        },
        2: {
            'kind': 'worker',
            'globus_comp_id': None, # or UUID
            'proxystore_id': None, # or UUID
            'children': [],
            'extra: None,
        },
    }
    ```
```
"""
