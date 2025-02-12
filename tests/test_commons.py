import pytest

from flight.commons import proportion_split


def test_valid_proportions():
    lst = list(range(10))
    assert proportion_split(lst, (0.5, 0.5)) == ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
    assert proportion_split(lst, (0.5, 0.2, 0.3)) == (
        [0, 1, 2, 3, 4],
        [5, 6],
        [7, 8, 9],
    )


def test_proportions_sum_not_one():
    with pytest.raises(ValueError):
        proportion_split([1, 2, 3], (0.5, 0.6))


def test_negative_proportions():
    with pytest.raises(ValueError):
        proportion_split([1, 2, 3], (-0.5, 1.5))


def test_more_proportions_than_elements():
    with pytest.raises(ValueError):
        proportion_split([1, 2], (0.5, 0.5, 0.5))
