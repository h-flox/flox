from collections import defaultdict
from typing import Any


def extend_dicts(
    *dicts: dict[Any, list[Any]], pure_dict: bool = True
) -> dict[Any, list[Any]] | defaultdict[Any, list[Any]]:
    """
    Takes some variable number of ``dict`` objects and will append them along each key.

    Args:
        *dicts (dict[Any, Any]): A variable number of `dict` objects.
        pure_dict (bool): If ``True``, then return a standard Python dict; otherwise return the ``defaultdict`` used
            during execution.

    Examples:
        >>> d1: dict[str, list] = {"name": ["Alice", "Bob"], "age": [18, 19]}
        >>> d2: dict[str, list] = {"name": ["Carol"], "age": [20]}
        >>> print(extend_dicts(d1, d2))
        {'name': ['Alice', 'Bob', 'Carol'], 'age': [18, 19, 20]}

    Error:
        A ``ValueError()`` is raised if there is a mismatch the keys among the passed in ``dict`` objects.
        A ``TypeError()`` is raised if values of dicts are not instances of a ``list``.

    Returns:
        Extended ``dict``.
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

    if pure_dict:
        return dict(new_dict)
    else:
        return new_dict
