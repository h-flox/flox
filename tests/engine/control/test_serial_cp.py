import pytest

from testing.fixtures import serial_controller


def fn_divide_by_zero(num: int | float) -> int | float:
    return num / 0


def fn_identity(num: int | float) -> int | float:
    return num


def fn_square(num: int | float) -> float:
    return float(num * num)


def test_valid_uses(serial_controller):
    for num in range(0, 100 + 1, 10):
        fut = serial_controller(fn_identity, num=num)
        assert fut.result() == num

        fut = serial_controller(fn_square, num=num)
        assert fut.result() == (num**2)


def test_invalid_uses(serial_controller):
    fut = serial_controller(fn_divide_by_zero, num=10)
    with pytest.raises(ZeroDivisionError):
        fut.result()
