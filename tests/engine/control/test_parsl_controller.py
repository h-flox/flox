# import pytest
#
# from testing.fixtures import parsl_controller
#
#
# def fn_divide_by_zero(num: int | float) -> float:
#     return num / 0
#
#
# def fn_identity(num: int | float) -> int | float:
#     return num
#
#
# def fn_square(num: int | float) -> float:
#     return float(num * num)
#
#
# def test_valid_uses(parsl_controller):
#     for num in range(0, 100 + 1, 10):
#         fut = parsl_controller(fn_identity, num=num)
#         assert fut.result() == num
#
#         fut = parsl_controller(fn_square, num=num)
#         assert fut.result() == (num**2)
#
#
# def test_invalid_uses(parsl_controller):
#     fut = parsl_controller(fn_divide_by_zero, num=10)
#     with pytest.raises(ZeroDivisionError):
#         fut.result()
