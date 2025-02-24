import pytest

from flight.fitter import simple_federated_fit


def test_federated_fit_without_failure():
    try:
        simple_federated_fit(None, 10)
        assert True
    except BaseException as err:
        pytest.fail(f"Unexpected error: {err}")


def test_federated_fit_event_hooks():
    pass  # TODO
