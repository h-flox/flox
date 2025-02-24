import itertools
import pytest

from flight.events import CoordinatorEvents, AggregatorEvents, WorkerEvents


def test_non_overlapping_event_names():
    starts = [
        CoordinatorEvents.STARTED,
        AggregatorEvents.STARTED,
        WorkerEvents.STARTED,
    ]
    for s1, s2 in itertools.permutations(starts, 2):
        assert s1 == s1
        assert s1 != s2
        assert s1.value == s2.value

    completes = [
        CoordinatorEvents.COMPLETED,
        AggregatorEvents.COMPLETED,
        WorkerEvents.COMPLETED,
    ]
    for c1, c2 in itertools.permutations(completes, 2):
        assert c1 == c1
        assert c1 != c2
        assert c1.value == c2.value
