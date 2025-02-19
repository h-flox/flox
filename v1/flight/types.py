import typing as t

Record: t.TypeAlias = dict[str, t.Any]
"""
A single list of key-value pairs (defined as `dict`s).
"""

P = t.ParamSpec("P")
"""
Generic parameter specification used throughout Flight.
"""

T = t.TypeVar("T")
"""
Generic type variable used throughout Flight.
"""
