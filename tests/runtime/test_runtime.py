from flight.runtime import Runtime


def double_fn(x):
    return x * 2


def square_fn(x):
    return x * x


def test_runtime_with_threads():
    runtime = Runtime.simple_setup(max_workers=1, exec_kind="thread")
    assert runtime.submit(double_fn, 10).result() == 20
    assert runtime.submit(square_fn, 10).result() == 100
    assert runtime.transfer(10) == 10


def test_runtime_with_processes():
    runtime = Runtime.simple_setup(max_workers=1, exec_kind="process")
    assert runtime.submit(double_fn, 10).result() == 20
    assert runtime.submit(square_fn, 10).result() == 100
    assert runtime.transfer(10) == 10
