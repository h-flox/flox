from flight.learning.parameters import parameters
from flight.strategies.commons.averaging import average_params, weighted_average_params


def test_average_params():
    # Params from `list[float]`.
    p0 = parameters([("a", [0.0]), ("b", [0.0])])
    p1 = parameters([("a", [5.0]), ("b", [10.0])])

    p = average_params([p0, p1])
    p.torch()

    assert p["a"] == 2.5
    assert p["b"] == 5.0

    # Params from `list[int]`.
    p0 = parameters([("a", [0]), ("b", [0])])
    p1 = parameters([("a", [5]), ("b", [10])])

    p = average_params([p0, p1])
    p.torch()

    assert p["a"] == 2.5
    assert p["b"] == 5.0

    # Params from `int`.
    # p0 = parameters([("a", 0), ("b", 0)])
    # p1 = parameters([("a", 5), ("b", 10)])
    #
    # p = average_params([p0, p1])
    # p.torch()
    #
    # assert p["a"] == 2.5
    # assert p["b"] == 5.0
    #
    # # Params from `float`.
    # p0 = parameters([("a", 0.0), ("b", 0.0)])
    # p1 = parameters([("a", 5.0), ("b", 10.0)])
    #
    # p = average_params([p0, p1])
    # p.torch()
    #
    # assert p["a"] == 2.5
    # assert p["b"] == 5.0


def test_weighted_average_params():
    # Params from `list[float]`.
    p0 = parameters([("a", [0.0]), ("b", [0.0])])
    p1 = parameters([("a", [5.0]), ("b", [10.0])])

    p = weighted_average_params([p0, p1], weights=[0.5, 0.5])
    assert p["a"] == 2.5
    assert p["b"] == 5.0

    p = weighted_average_params([p0, p1], weights=[1, 1])
    assert p["a"] == 2.5
    assert p["b"] == 5.0

    p = weighted_average_params([p0, p1], weights=[1, 0])
    assert p["a"] == 0.0
    assert p["b"] == 0.0

    p = weighted_average_params([p0, p1], weights=[0, 1])
    assert p["a"] == 5.0
    assert p["b"] == 10.0

    # Params from `list[int]`.
    # p0 = parameters([("a", [0]), ("b", [0])])
    # p1 = parameters([("a", [5]), ("b", [10])])
    #
    # p = weighted_average_params([p0, p1])
    #
    # assert p["a"] == 2.5
    # assert p["b"] == 5.0
    #
    # # Params from `int`.
    # p0 = parameters([("a", 0), ("b", 0)])
    # p1 = parameters([("a", 5), ("b", 10)])
    #
    # p = weighted_average_params([p0, p1])
    #
    # assert p["a"] == 2.5
    # assert p["b"] == 5.0
    #
    # # Params from `float`.
    # p0 = parameters([("a", 0.0), ("b", 0.0)])
    # p1 = parameters([("a", 5.0), ("b", 10.0)])
    #
    # p = average_params([p0, p1])
    # p.torch()
    #
    # assert p["a"] == 2.5
    # assert p["b"] == 5.0
