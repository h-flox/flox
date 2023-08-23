from __future__ import annotations

from collections import defaultdict
from typing import Any


def extend_dicts(*dicts: dict[Any, Any]):
    """

    Args:
        *dicts (dict[Any, Any]): A variable number of `dict` objects.

    Examples:
        >>> d1 = {"name": ["Alice", "Bob"], "age": [18, 19]}
        >>> d2 = {"name": ["Carol"], "age": [20]}
        >>> print(extend_dicts(d1, d2))
        defaultdict(<class 'list'>, {'name': ['Alice', 'Bob', 'Carol'], 'age': [18, 19, 20]})

    Returns:

    """
    num_keys = None
    new_dict = defaultdict(list)

    for d in dicts:
        assert isinstance(d, dict)

        if num_keys is None:
            num_keys = len(d)

        if len(d) != num_keys:
            raise ValueError()

        for key, val in d.items():
            assert isinstance(val, list)
            new_dict[key].extend(val)

    return new_dict
