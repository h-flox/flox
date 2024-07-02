import sys
import typing as t
from collections import defaultdict

from numpy.random import Generator, default_rng


class _ProbSkipBlock(Exception):
    pass


class Probability:
    """
    A utility class for executing code blocks probabilistically using a context manager.

    A simple example can be seen below:
    >>> with Probability(p=0.5):
    >>>     print("This will be printed with a 50% likelihood.")

    If you wish to have reproducible executions, then it is recommended that you generate
    a `Generator` from `numpy` using `numpy.random.default_rng`. This can then be passed
    into the `Probability` class. An example is below:
    >>> gen = default_rng(1)
    >>> with Probability(p=0.5, rng=gen):
    >>>     print("This will be printed with a 50% likelihood.")

    While, you can pass an integer seed directly into the
    class instance, this is not encouraged. Because running that separate times will fix
    all iterations to the same outcome.
    """

    def __init__(self, p: float, rng: t.Optional[Generator | int] = None):
        if not 0.0 <= p <= 1.0:
            raise ValueError(
                "Illegal value for probability `p`. Must be in range [0.0, 1.0]."
            )

        if not isinstance(rng, Generator):
            rng = default_rng(rng)

        self.skip = bool(p <= rng.random())

    def __enter__(self):
        if self.skip:
            sys.settrace(lambda *args, **keys: None)  # noqa
            frame = sys._getframe(1)  # noqa
            frame.f_trace = self.trace

    def trace(self, frame, event, arg):
        raise _ProbSkipBlock()

    def __exit__(self, _type, value, traceback):
        if _type is None:
            return
        if issubclass(_type, _ProbSkipBlock):
            return True


def extend_dicts(
    *dicts: dict[t.Any, list[t.Any]], pure_dict: bool = True
) -> dict[t.Any, list[t.Any]] | defaultdict[t.Any, list[t.Any]]:
    """
    Takes some variable number of ``dict`` objects and will append them along each key.

    Args:
        *dicts (dict[Any, Any]): A variable number of `dict` objects.
        pure_dict (bool): If ``True``, then return a standard Python dict; otherwise
            return the ``defaultdict`` used during execution.

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
