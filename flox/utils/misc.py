from __future__ import annotations

from collections import defaultdict
from typing import Any


def extend_dicts(
    *dicts: dict[Any, list[Any]], pure: bool = True
) -> dict[Any, list[Any]]:
    """
    Takes some variable number of ``dict`` objects and will append them along each key.

    Args:
        *dicts (dict[Any, Any]): A variable number of `dict` objects.
        pure (bool): Returns a standard ``dict`` if True; a ``defaultdict`` if False.

    Examples:
        >>> d1: dict[str, list] = {"name": ["Alice", "Bob"], "age": [18, 19]}
        >>> d2: dict[str, list] = {"name": ["Carol"], "age": [20]}
        >>> print(extend_dicts(d1, d2))
        defaultdict(<class 'list'>, {'name': ['Alice', 'Bob', 'Carol'], 'age': [18, 19, 20]})

    Error:
        A ``ValueError()`` is raised if there is a mismatch the keys among the passed in ``dict`` objects.
        A ``TypeError()`` is raised if values of dicts are not instances of a ``list``.

    Returns:
        Extended ``dict`` (or ``defaultdict``) object.
    """
    key_set = None
    new_dict = defaultdict(list)

    for d in dicts:
        assert isinstance(d, dict)

        if key_set is None:
            key_set = set(d.keys())

        for key, val in d.items():
            if not isinstance(val, list):
                raise TypeError("Values in dicts must be instance of ``list``.")
            if key not in key_set:
                raise ValueError("Inconsistent keys across passed in ``dicts``.")
            new_dict[key].extend(val)

    return dict(new_dict) if pure else new_dict
