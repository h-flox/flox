import pytest

from flight.engine.control.serial import SerialController


@pytest.fixture
def controller() -> SerialController:
    return SerialController()


class TestSerialControlPane:
    @staticmethod
    def divide_by_zero(num: int | float) -> int | float:
        return num / 0

    @staticmethod
    def identity(num: int | float) -> int | float:
        return num

    @staticmethod
    def square(num: int | float) -> float:
        return float(num * num)

    def test_valid_uses(self, controller):
        for num in range(0, 100 + 1, 10):
            fut = controller(self.identity, num=num)
            assert fut.result() == num

            fut = controller(self.square, num=num)
            assert fut.result() == (num**2)

    def test_invalid_uses(self, controller):
        fut = controller(self.divide_by_zero, num=10)
        with pytest.raises(ZeroDivisionError):
            fut.result()
