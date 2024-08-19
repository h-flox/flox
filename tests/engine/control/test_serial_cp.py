import pytest

from flight.engine.control.serial import SerialCP


@pytest.fixture
def controller() -> SerialCP:
    return SerialCP()


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
            fut = controller(self.identity, num)
            assert fut.result() == num

            fut = controller(self.square, num)
            assert fut.result() == (num**2)

    def test_invalid_uses(self, controller):
        fut = controller(self.divide_by_zero, 10)
        fut = controller(self.divide_by_zero, 10)
        fut.result()
