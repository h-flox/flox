from __future__ import annotations

import typing as t

import numpy as np

if t.TYPE_CHECKING:
    T = t.TypeVar("T")


def proportion_split(
    seq: t.Sequence[T], proportion: t.Sequence[float, ...]
) -> tuple[t.Sequence[T], ...]:
    """
    Split a sequence into multiple sequences based on proportions.

    Args:
        seq (Sequence[T]): Sequence to split.
        proportion (t.Sequence[float, ...]): Proportions to split the sequence.

    Returns:
        Sequences split based on the proportions. The number of sequences returned is
            equal to the length of the `proportion` argument.

    Examples:
        >>> lst = list(range(10))
        >>> lst
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> proportion_split(lst, (0.5, 0.5))
        ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
        >>> proportion_split(lst, (0.5, 0.2, 0.3))
        ([0, 1, 2, 3, 4], [5, 6], [7, 8, 9])

    Throws:
        - `ValueError`: If the values in `proportion` argument do not sum to 1.
    """
    if sum(proportion) != 1:
        raise ValueError("Proportions must sum to 1.")

    total = len(seq)
    splits = np.cumsum(np.array(proportion) * total).astype(int)
    splits = np.append(np.array([0]), splits)
    return tuple(seq[splits[i - 1] : splits[i]] for i in range(1, len(splits)))  # noqa


def random_generator(
    rng: np.random.Generator | int | None = None,
) -> np.random.Generator:
    """
    Create a random number generator.

    Args:
        rng (numpy.random.Generator | int | None): Random number generator.

    Returns:
        numpy.random.Generator: Random number generator.

    Notes:
        What is returned by this function depends on what is given to the `rng` arg:

        1. If `rng` is a `numpy.random.Generator`, it is returned as is.
        2. If `rng` is an integer, it is used to seed the random number generator.
        3. If `rng` is `None`, then a pseudorandom  random number generator is
            returned using `numpy.random.default_rng(None)`.
    """
    if rng is None:
        return np.random.default_rng()
    elif isinstance(rng, int):
        return np.random.default_rng(rng)
    else:
        return rng
